import sys
sys.path.append('..')

import torch
from torch import nn
from data.vocab import Vocab
from models.light import LiGhTPredictor as LiGhT, NodeSelfAttention
from models.graphgps import GraphGPS
from utils.function import load_config


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def get_predictor(dim_emb, n_tasks, n_layers, predictor_drop, device, d_hidden_feats=None):
    if n_layers == 1:
        predictor = nn.Linear(dim_emb, n_tasks)
    else:
        predictor = nn.ModuleList()
        predictor.append(nn.Linear(dim_emb, d_hidden_feats))
        predictor.append(nn.Dropout(predictor_drop))
        predictor.append(nn.GELU())
        for _ in range(n_layers-2):
            predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
        predictor.append(nn.Linear(d_hidden_feats, n_tasks))
        predictor = nn.Sequential(*predictor)
    predictor.apply(lambda module: init_params(module))
    return predictor.to(device)


def get_node_attn(dim_emb, dropout, device):
    node_attn = NodeSelfAttention(dim_emb, dropout)
    node_attn.apply(lambda module: init_params(module))
    return node_attn.to(device)


def get_model(args):
    config = load_config(args)
    vocab = Vocab()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.backbone == "light":
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=1191,
            d_md_feats=1613,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=args.dropout,
            feat_drop=args.dropout,
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
            input_drop=0,
            attn_drop=args.dropout,
            local_gnn=config['local_gnn']
        ).to(device)
    else:
        raise ValueError("Backbone is not implemented")
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f'{args.model_path}').items()})
    if args.mode == "freeze":
        for p in model.parameters():
            p.requires_grad = False
    model.predictor = get_predictor(dim_emb=config['d_g_feats'] * 3, n_tasks=1, n_layers=2, predictor_drop=args.dropout, device=device, d_hidden_feats=256)
    model.node_attn = get_node_attn(dim_emb=config['d_g_feats'], dropout=args.dropout, device=device)

    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor
    del model.cl_projector

    return model