import argparse
from multiprocessing import cpu_count
import pandas as pd
from dgl import save_graphs
import pickle
import numpy as np
import dgl.backend as F
import torch
from tqdm import tqdm
from scipy import sparse as sp
import warnings

import sys
sys.path.append('..')
from data.vocab import Vocab
from models.light import LiGhTPredictor as LiGhT
from models.graphgps import GraphGPS
from utils.function import load_config
from utils.features import precompute_features
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--backbone", type=str, default="light")
    parser.add_argument("--model_path", type=str, default="../checkpoints/pretrained/light/base.pth")
    parser.add_argument("--data_path", type=str, default='../datasets/')
    parser.add_argument("--no_prompt", action='store_true')

    args = parser.parse_args()
    return args


def get_model(args):
    config = load_config(args)
    vocab = Vocab()
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
            attn_drop=0,
            feat_drop=0,
            n_node_types=vocab.vocab_size
        )
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
        )
    else:
        raise ValueError("Backbone is not implemented")

    model = model.to(torch.device(args.device))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.model_path).items()})
    return model


def prepare_dataset(args):
    with open("../datasets/pretrain/scaler_all.pkl", 'rb') as file:
        scaler = pickle.load(file)

    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/graphs.pkl"
    smiless = df.smiles.values.tolist()
    task_names = df.columns.drop(['smiles']).tolist()

    model = get_model(args)

    print('Precomputing features')
    feat_cache = precompute_features(smiless, units=(3, 6, 9), workers=args.workers)

    if args.backbone == "light":
        from data.smiles2g_light import smiles_to_graph_with_prompt, smiles_to_graph_without_prompt
    elif args.backbone == "graphgps":
        from data.smiles2g_graphgps import smiles_to_graph_with_prompt, smiles_to_graph_without_prompt
    else:
        raise ValueError

    print('constructing graphs')
    graphs = []
    for smiles in tqdm(smiless, total=len(smiless), ncols=100):
        if args.no_prompt:
            g = smiles_to_graph_without_prompt(smiles, n_virtual_nodes=2)
        else:
            g = smiles_to_graph_with_prompt(smiles, model, scaler, args.device, feat_cache, n_virtual_nodes=2)
        graphs.append(g)

    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    assert len(valid_graphs) == len(graphs)
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print('saving graphs')
    print(f'graphs length: {len(valid_ids)}')
    save_graphs(cache_file_path, valid_graphs, labels={'labels': labels})

    print('extracting expert knowledge')
    fp_list = []
    des_list = []
    for smiles in smiless:
        fp, md = feat_cache.get((smiles, 3), (None, None))
        fp_list.append(fp)
        des_list.append(md.astype(np.float32))

    fp_list = np.array(fp_list, dtype=np.float32)
    print(f"fp shape: {fp_list.shape}")
    fp_sp_mat = sp.csc_matrix(fp_list)
    sp.save_npz(f"{args.data_path}/{args.dataset}/maccs_ecfp.npz", fp_sp_mat)

    des = np.array(des_list)
    des_norm = scaler.transform(des)
    print(f"des shape: {des.shape}")
    np.savez_compressed(f"{args.data_path}/{args.dataset}/polymer_descriptors.npz", md=des_norm)


if __name__ == '__main__':
    args = parse_args()
    prepare_dataset(args)


