import sys
sys.path.append('..')
from data.smiles2g import PolyGraphBuilder
from utils.features import precompute_features
import argparse
import os
import pickle
import pandas as pd
from dgl import save_graphs
import numpy as np
import dgl.backend as F
from tqdm import tqdm
from scipy import sparse as sp
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--model_path", type=str, default="../checkpoints/pretrained/light/base.pth")
    parser.add_argument("--max_prompt", type=int, default=10)
    parser.add_argument("--data_path", type=str, default='../datasets/')
    parser.add_argument("--n_jobs", type=int, default=4)

    args = parser.parse_args()
    return args


def prepare_dataset(args):
    with open("../datasets/pretrain/scaler_all.pkl", 'rb') as file:
        scaler = pickle.load(file)

    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/graphs.pkl"
    smiless = df.smiles.values.tolist()
    task_names = ['PCE_ave']

    monomer_list = []
    for smiles in smiless:
        monomer_list.append([smiles])

    print('Precomputing features')
    unique_smiless = list(set(smiless))
    feat_cache = precompute_features(unique_smiless, units=(3,), workers=args.n_jobs)

    print('Constructing graphs')
    graphs = []
    builder = PolyGraphBuilder(args, scaler, n_local_nodes=2, n_global_nodes=1)
    for smiles_lst in tqdm(monomer_list, total=len(monomer_list), ncols=100):
        g = builder.build(smiles_lst, feat_cache)
        graphs.append(g)

    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)

    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(_label_values.astype(np.float32))[valid_ids]
    print(f'Saving graphs len(graphs)={len(valid_ids)}')
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

    print('extracting global state')
    global_state = df.iloc[:, 1:-1].values
    print(f"global state shape: {global_state.shape}")
    np.savez_compressed(f"{args.data_path}/{args.dataset}/global_state.npz", gs=global_state)


if __name__ == '__main__':
    args = parse_args()
    prepare_dataset(args)


