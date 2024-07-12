import sys
sys.path.append('..')

from torch import nn
from data.vocab import Vocab
from models.light import LiGhTPredictor as LiGhT
from models.graphgps import GraphGPS
from utils.function import load_config


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_model(args, device):
    config = load_config(args)
    vocab = Vocab()
    if args.backbone == "light":
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_cl_feats=config['d_cl_feats'],
            d_fp_feats=1191,
            d_md_feats=1613,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=config['input_drop'],
            attn_drop=config['attn_drop'],
            feat_drop=config['feat_drop'],
            n_node_types=vocab.vocab_size
        ).to(device)
    elif args.backbone == "graphgps":
        model = GraphGPS(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=1191,
            d_md_feats=1613,
            n_heads=config['n_heads'],
            n_layers=config['n_mol_layers'],
            n_node_types=vocab.vocab_size,
            input_drop=config['input_drop'],
            attn_drop=config['attn_drop'],
            local_gnn=config['local_gnn']
        ).to(device)
    else:
        raise ValueError("Backbone is not implemented")

    return model
