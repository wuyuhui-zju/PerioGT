"""
The implementation of LiGhT refers to its original article and code
"KPGT: Knowledge-Guided Pre-training of Graph Transformer or Molecular Property Prediction"
by Han Li, KDD 2022, https://arxiv.org/abs/2206.03364
"""

import torch
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import numpy as np

from data.smiles2g import VIRTUAL_ATOM_FEATURE_PLACEHOLDER, VIRTUAL_BOND_FEATURE_PLACEHOLDER


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation, d_hidden_feats=d_out_feats * 4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x


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


class TripletTransformer(nn.Module):
    def __init__(self,
                 d_feats,
                 d_hpath_ratio,
                 path_length,
                 n_heads,
                 n_ffn_dense_layers,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU()):
        super(TripletTransformer, self).__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, activation)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, triplet_h, dist_attn, path_attn):
        g = g.local_var()
        new_triplet_h = self.attention_norm(triplet_h)
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)  # Size: [3, num_nodes, 4, 32]
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]  # Size: [num_nodes, 4, 32]
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + path_attn.reshape(
            len(g.edata['node_attn']), -1, 1)  # Size: [num_edges, 4, 1]
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))

        g.ndata['hv'] = v.view(-1, self.d_feats)  # Size: [num_nodes, 128]
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats // self.n_heads)) * g.edata['sa']).view(-1, self.d_feats)

        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        return self.node_out_layer(triplet_h, g.ndata['agg_h'])

    def _device(self):
        return next(self.parameters()).device


class LiGhT(nn.Module):
    def __init__(self,
                 d_g_feats,
                 d_hpath_ratio,
                 path_length,
                 n_mol_layers=2,
                 n_heads=4,
                 n_ffn_dense_layers=4,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU()):
        super(LiGhT, self).__init__()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats // d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Distance Attention
        self.path_len_emb = nn.Embedding(path_length + 1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(self.d_g_feats, self.d_g_feats),
            activation,
            nn.Linear(self.d_g_feats, n_heads)
        )
        # Path Attention
        self.trip_fortrans = nn.ModuleList([
            MLP(d_g_feats, self.d_trip_path, 2, activation) for _ in range(self.path_length)
        ])
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            activation,
            nn.Linear(self.d_trip_path, n_heads)
        )
        # Molecule Transformer Layers
        self.mol_T_layers = nn.ModuleList([
            TripletTransformer(d_g_feats, d_hpath_ratio, path_length, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
                               activation) for _ in range(n_mol_layers)
        ])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def _featurize_path(self, g, path_indices):
        mask = (path_indices[:, :] >= 0).to(torch.int32)
        path_feats = torch.sum(mask, dim=-1)
        path_feats = self.path_len_emb(path_feats)
        path_feats[g.edata['vp'] == 1] = self.virtual_path_emb.weight  # virtual path
        path_feats[g.edata['sl'] == 1] = self.self_loop_emb.weight  # self loop
        return path_feats

    def _init_path(self, g, triplet_h, path_indices):
        g = g.local_var()
        path_indices[path_indices < -99] = -1
        path_h = []
        for i in range(self.path_length):
            path_h.append(torch.cat([self.trip_fortrans[i](triplet_h), torch.zeros(size=(1, self.d_trip_path)).to(self._device())], dim=0)[path_indices[:, i]])
        path_h = torch.stack(path_h, dim=-1)  # Size: [n_edges, d_trip_path, path_length]
        mask = (path_indices >= 0).to(torch.int32)
        path_size = torch.sum(mask, dim=-1, keepdim=True)
        path_h = torch.sum(path_h, dim=-1) / path_size
        return path_h  # Size: [n_edges, d_trip_path]

    def forward(self, g, triplet_h):
        path_indices = g.edata['path']
        dist_h = self._featurize_path(g, path_indices)
        path_h = self._init_path(g, triplet_h, path_indices)
        dist_attn, path_attn = self.dist_attn_layer(dist_h), self.path_attn_layer(path_h)
        for i in range(self.n_mol_layers):
            triplet_h = self.mol_T_layers[i](g, triplet_h, dist_attn, path_attn)
        return triplet_h

    def _device(self):
        return next(self.parameters()).device


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
            d_fp_feats,
            d_md_feats,
            activation=nn.GELU()):
        super(TripletEmbedding, self).__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(self, node_h, edge_h, fp, md, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        triplet_h[indicators == 1] = self.fp_proj(fp)
        triplet_h[indicators == 2] = self.md_proj(md)
        return triplet_h


class TripletEmbeddingCopoly(nn.Module):
    def __init__(
            self,
            d_g_feats,
            d_fp_feats,
            d_md_feats,
            activation=nn.GELU()):
        super(TripletEmbeddingCopoly, self).__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(self, node_h, edge_h, fp_1, md_1, fp_2, md_2, ratio, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)

        triplet_h[indicators==1] = self.fp_proj(fp_1)
        triplet_h[indicators==2] = self.md_proj(md_1)
        triplet_h[indicators == 3] = self.embedding(ratio[:, 0].long())

        triplet_h[indicators==4] = self.fp_proj(fp_2)
        triplet_h[indicators==5] = self.md_proj(md_2)
        triplet_h[indicators == 6] = self.embedding(ratio[:, 1].long())

        triplet_h[indicators==7] = self.global_node_emb(ratio[:, 2].long())
        return triplet_h


class NodeSelfAttention(nn.Module):
    def __init__(self, dim_emb, dropout=0.1):
        super().__init__()
        self.dim_emb = dim_emb
        # self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_emb))
        self.self_attn_1 = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=8, batch_first=True)
        self.self_attn_2 = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=8, batch_first=True)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)
        self.norm_1 = nn.LayerNorm(dim_emb)
        self.norm_2 = nn.LayerNorm(dim_emb)
        self.linear_1 = nn.Linear(dim_emb, dim_emb)
        self.linear_2 = nn.Linear(dim_emb, dim_emb)
        self.linear_3 = nn.Linear(dim_emb, dim_emb)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.virutal_node_emb = nn.Embedding(1, dim_emb)

    def forward(self, x, triplet_h, indicators):
        x[indicators == 3][:, 0, :] = self.virutal_node_emb.weight
        x[indicators == 6][:, 0, :] = self.virutal_node_emb.weight
        x[indicators == 7][:, 0, :] = self.virutal_node_emb.weight

        # x 的形状是 [num_node, len, dim_emb]
        num_node, seq_len, _ = x.size()

        # 为每个节点复制cls_token
        cls_tokens = self.cls_token.expand(num_node, -1, -1)

        # 将cls_token添加到序列的前面
        x = torch.cat((cls_tokens, x), dim=1)  # 新的形状是 [num_node, len+1, dim_emb]

        # 生成一个mask，以忽略padding的token
        mask = (x.sum(dim=-1) == 0)  # 计算哪些位置是padding（在最后一个维度上求和）
        hidden_state, _ = self.self_attn_1(x, x, x, key_padding_mask=mask)
        hidden_state = self.linear_1(hidden_state)
        hidden_state = self.dropout_1(hidden_state)
        x = self.norm_1(hidden_state + x)

        hidden_state, _ = self.self_attn_2(x, x, x, key_padding_mask=mask)
        hidden_state = self.linear_2(hidden_state)
        hidden_state = self.dropout_2(hidden_state)
        hidden_state = self.norm_2(hidden_state + x)

        cls_output = hidden_state[:, 0, :]  # 形状是 [num_node, dim_emb]
        cls_output = self.linear_3(cls_output)

        return triplet_h + self.alpha * cls_output


class LiGhTPredictor(nn.Module):
    def __init__(self,
                 d_node_feats=40,
                 d_edge_feats=12,
                 d_g_feats=128,
                 d_cl_feats=256,
                 d_fp_feats=512,
                 d_md_feats=200,
                 d_hpath_ratio=1,
                 n_mol_layers=2,
                 path_length=5,
                 n_heads=4,
                 n_ffn_dense_layers=2,
                 input_drop=0.,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation=nn.GELU(),
                 n_node_types=1,
                 readout_mode='mean'
                 ):
        super(LiGhTPredictor, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Model
        self.model = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop,
            activation
        )
        # Predict
        # self.node_predictor = nn.Linear(d_g_feats, n_node_types)
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
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h[g.ndata['mask'] == 1] = self.mask_emb.weight
        # Model
        triplet_h = self.model(g, triplet_h)
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

    def forward_tune(self, g, fp_1, md_1, fp_2, md_2, ratio):
        indicators = g.ndata['vavn'] # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp_1, md_1, fp_2, md_2, ratio, indicators)
        triplet_h = self.node_attn(g.ndata['prompt'], triplet_h, indicators)

        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn_1 = triplet_h[indicators == 1]
        md_vn_1 = triplet_h[indicators == 2]
        ratio_vn_1 = triplet_h[indicators == 3]

        fp_vn_2 = triplet_h[indicators == 4]
        md_vn_2 = triplet_h[indicators == 5]
        ratio_vn_2 = triplet_h[indicators == 6]

        copolym_type = triplet_h[indicators == 7]

        fp_vn = (fp_vn_1 + fp_vn_2) / 2
        md_vn = (md_vn_1 + md_vn_2) / 2
        ratio_vn = (ratio_vn_1 + ratio_vn_2) / 2

        g.remove_nodes(np.where(indicators.detach().cpu().numpy()>=1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, ratio_vn, copolym_type, readout],dim=-1)
        #Predict
        return self.predictor(g_feats)

    def generate_node_emb(self, g, fp, md):
        bids = g.ndata['bid']  # -1 indicates triplet contain * atom or virtual nodes
        indicators = g.ndata['vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)

        return triplet_h, bids