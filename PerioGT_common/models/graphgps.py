"""GraphGPS was introduced in the reference: Recipe for a General, powerful, scalable graph transformer"""

import torch
import torch.nn.functional as F
from models.pnaconv import PNAConv

from dgl.nn.pytorch.conv import GINConv
from torch import nn
import dgl
import numpy as np
from typing import Optional, Tuple
from torch import Tensor

from data.smiles2g_light import VIRTUAL_ATOM_FEATURE_PLACEHOLDER, VIRTUAL_BOND_FEATURE_PLACEHOLDER


def to_dense_batch(
        x: Tensor,
        graph_nodes: Optional[Tensor] = None,
        max_num_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_nodes: int = 0,
) -> Tuple[Tensor, Tensor]:
    batch = torch.zeros((num_nodes), dtype=torch.int64, device=x.device)
    batch[graph_nodes[:-1].cumsum(0)] = 1
    batch = batch.cumsum(0)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    cum_nodes = torch.cat([batch.new_zeros(1), graph_nodes.cumsum(dim=0)])

    if max_num_nodes is None:
        max_num_nodes = int(graph_nodes.max())

    # cal the index of each nodes
    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    # the output size
    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, 0)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats


class GraphGPSLayer(nn.Module):
    def __init__(
            self, hidden_size, num_heads, local_gnn, attn_type, dropout=0., batch_norm=True
    ):
        super().__init__()
        self.activation = nn.ReLU()

        # Local GNN model
        self.local_gnn = local_gnn
        if self.local_gnn == "GIN":
            gin_nn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, hidden_size),
            )
            self.local_model = GINConv(gin_nn, aggregator_type="sum")
        elif self.local_gnn == "PNA":
            self.local_model = PNAConv(hidden_size, hidden_size, ['mean', 'max', 'sum'], ['identity', 'amplification'],
                                       2.5)
        else:
            raise ValueError(f"Unsupported Local GNN model {self.local_gnn}")

        # Global attention transformer model
        self.attn_type = attn_type
        if attn_type == None:
            self.global_attn = None
        elif attn_type == "Transformer":
            self.global_attn = torch.nn.MultiheadAttention(
                hidden_size, num_heads, dropout, batch_first=True
            )
        else:
            raise ValueError(
                f"Unsupported Global Attention Transformer model {attn_type}"
            )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm_local = nn.BatchNorm1d(hidden_size)
            self.norm_attn = nn.BatchNorm1d(hidden_size)
            self.norm_out = nn.BatchNorm1d(hidden_size)

        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, g, h):
        h_in = h
        # Local MPNN
        h_local = self.local_model(g, h)
        h_local = self.dropout_local(h_local)
        h_local = h_in + h_local
        if self.batch_norm:
            h_local = self.norm_local(h_local)

        # Multi-head attention
        h_attn = self.attn_block(g, h)
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in + h_attn
        if self.batch_norm:
            h_attn = self.norm_attn(h_attn)

        # Combine the local and global outputs
        h = h_local + h_attn
        h = h + self.FFN2(F.relu(self.FFN1(h)))
        if self.batch_norm:
            h = self.norm_out(h)
        return h

    def attn_block(self, g, h):
        h_dense, mask = to_dense_batch(
            h, g.batch_num_nodes(), batch_size=g.batch_size, num_nodes=g.num_nodes()
        )
        x = self.global_attn(
            h_dense, h_dense, h_dense, key_padding_mask=~mask, need_weights=False
        )[0]
        h_attn = x[mask]

        return h_attn


class AtomEmbedding(nn.Module):
    def __init__(
            self,
            d_atom_feats,
            d_g_feats,
            input_drop):
        super(AtomEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, pair_node_feats, indicators):
        pair_node_h = self.in_proj(pair_node_feats)  # [num_nodes, 2, d_g_feats]
        pair_node_h[indicators == VIRTUAL_ATOM_FEATURE_PLACEHOLDER, 1, :] = self.virtual_atom_emb.weight  # .half()
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)  # [num_nodes, d_g_feats]


class BondEmbedding(nn.Module):
    def __init__(
            self,
            d_bond_feats,
            d_g_feats,
            input_drop):
        super(BondEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, edge_feats, indicators):
        edge_h = self.in_proj(edge_feats)
        edge_h[indicators == VIRTUAL_BOND_FEATURE_PLACEHOLDER] = self.virutal_bond_emb.weight  # .half()
        return self.input_dropout(edge_h)


class TripletEmbedding(nn.Module):
    def __init__(
            self,
            d_g_feats,
            pe_feats,
            d_fp_feats,
            d_md_feats,
            activation):
        super(TripletEmbedding, self).__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats - pe_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats - pe_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats - pe_feats, 2, activation)

    def forward(self, node_h, edge_h, fp, md, pe, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        triplet_h[indicators == 1] = self.fp_proj(fp)
        triplet_h[indicators == 2] = self.md_proj(md)
        return torch.cat([triplet_h, pe], dim=1)


class GraphGPS(nn.Module):
    def __init__(self,
                 d_node_feats=40,
                 d_edge_feats=12,
                 d_g_feats=128,
                 pe_feats=28,
                 d_cl_feats=256,
                 d_fp_feats=512,
                 d_md_feats=200,
                 n_heads=4,
                 n_layers=5,
                 activation=nn.ReLU(),
                 n_node_types=1,
                 input_drop=0.,
                 attn_drop=0.,
                 readout_mode='mean',
                 local_gnn='GIN',
                 global_attn='Transformer',
                 ):
        super(GraphGPS, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, pe_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Model
        self.layers = nn.ModuleList(
            [GraphGPSLayer(
                d_g_feats,
                n_heads,
                local_gnn,
                global_attn,
                dropout=attn_drop
            ) for _ in range(n_layers)]
        )
        # Predict
        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_md_feats)
        )
        self.cl_projector = nn.Sequential(
            nn.Linear(d_g_feats * 3, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_cl_feats)
        )

        self.apply(lambda module: init_params(module))

    def forward(self, g, fp, md):
        indicators = g.ndata['vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, g.ndata['PE'], indicators)
        triplet_h[g.ndata['mask'] == 1] = self.mask_emb.weight
        # Model
        for layer in self.layers:
            triplet_h = layer(g, triplet_h)

        # CL
        with g.local_scope():
            g.ndata['ht'] = triplet_h
            fp_vn = triplet_h[indicators == 1]
            md_vn = triplet_h[indicators == 2]
            g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
            readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
            g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return self.node_predictor(triplet_h[g.ndata['mask'] >= 1]), self.fp_predictor(
            triplet_h[indicators == 1]), self.md_predictor(triplet_h[indicators == 2]), self.cl_projector(g_feats)

    def forward_tune(self, g, fp, md):
        indicators = g.ndata['vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, g.ndata['PE'], indicators)
        # Model
        for layer in self.layers:
            triplet_h = layer(g, triplet_h)

        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return self.predictor(g_feats)

    def generate_fps(self, g, fp, md):
        indicators = g.ndata['vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats


if __name__ == "__main__":
    pass