import os
import argparse
from multiprocessing import cpu_count
import pandas as pd
from dgl import save_graphs
import pickle
import numpy as np
import dgl.backend as F
from tqdm import tqdm
from scipy import sparse as sp

import sys
sys.path.append('..')
from utils.features import precompute_features

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_prompt", type=int, default=10)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument("--n_jobs", type=int, default=max(1, cpu_count()-4))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_suffix = "" if args.use_prompt else "_feat"
    cache_file_path = f"{args.data_path}/{args.dataset}/graphs{cache_file_suffix}.pkl"
    smiless = df.smiles.values.tolist()
    task_names = df.columns.drop(['smiles']).tolist()
    with open("../datasets/pretrain/scaler_all.pkl", 'rb') as file:
        scaler = pickle.load(file)

    print('Precomputing features')
    feat_cache = precompute_features(smiless, units=(3,), workers=args.n_jobs)

    print('Constructing graphs')
    if args.backbone == "light":
        from data.smiles2g_light import PolyGraphBuilderFinetune
        builder = PolyGraphBuilderFinetune(args, scaler)
    elif args.backbone == "graphgps":
        from data.smiles2g_graphgps import PolyGraphBuilderFinetune
        builder = PolyGraphBuilderFinetune(args, scaler)
    else:
        raise ValueError("Backbone is not implemented")

    graphs = []
    for smiles in tqdm(smiless, total=len(smiless), ncols=100):
        g = builder.build(smiles, feat_cache)
        graphs.append(g)

    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    assert len(valid_graphs) == len(graphs), "Error: found None in graphs"
    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print('Saving graphs')
    print(f'graphs length: {len(valid_ids)}')
    save_graphs(cache_file_path, valid_graphs, labels={'labels': labels})

    print('Extracting physicochemical features')
    fp_list = []
    des_list = []
    for smiles in smiless:
        fp, md = feat_cache.get((smiles, 3), (None, None))
        fp_list.append(fp)
        des_list.append(md.astype(np.float32))

    fp_path = f"{args.data_path}/{args.dataset}/maccs_ecfp.npz"
    des_path = f"{args.data_path}/{args.dataset}/polymer_descriptors.npz"
    if not os.path.exists(fp_path):
        fp_list = np.array(fp_list, dtype=np.float32)
        print(f"fp shape: {fp_list.shape}")
        fp_sp_mat = sp.csc_matrix(fp_list)
        sp.save_npz(fp_path, fp_sp_mat)
    else:
        print(f"Fingerprint file already exists: {fp_path}")

    if not os.path.exists(des_path):
        des_array = np.array(des_list)
        des_array = scaler.transform(des_array).astype(np.float32)
        print(f"des shape: {des_array.shape}")
        np.savez_compressed(des_path, md=des_array)
    else:
        print(f"Descriptor file already exists: {des_path}")
