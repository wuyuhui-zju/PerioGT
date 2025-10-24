import sys
sys.path.append('..')
from data.smiles2g import PolyGraphBuilderCopolym
from utils.features import precompute_features
import os
import argparse
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
    parser.add_argument("--max_prompt", type=int, default=5)
    parser.add_argument("--data_path", type=str, default='../datasets/')
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args


def preprocess_dataset(args):
    with open("../datasets/pretrain/scaler_all.pkl", 'rb') as file:
        scaler = pickle.load(file)

    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/graphs.pkl"
    smiless_1 = df["smiles_A"].values.tolist()
    smiless_2 = df["smiles_B"].values.tolist()
    task_names = ['value']

    monomer_list = []
    for smiles_1, smiles_2 in zip(smiless_1, smiless_2):
        monomer_list.append([smiles_1, smiles_2])

    print('Precomputing features')
    combined = smiless_1 + smiless_2
    unique_smiless = list(set(combined))
    feat_cache = precompute_features(unique_smiless, units=(3, 6, 9), workers=args.n_jobs)

    print('Constructing graphs')
    prompt_dict = {}
    builder = PolyGraphBuilderCopolym(args, scaler, n_local_nodes=3, n_global_nodes=1)
    for smiles in tqdm(unique_smiless, total=len(unique_smiless), ncols=100):
        prompt = builder.smiles_to_prompt(smiles, feat_cache)
        prompt_dict[smiles] = prompt

    graphs = []
    for smiles_lst in tqdm(monomer_list, total=len(monomer_list), ncols=100):
        prompt_1 = prompt_dict[smiles_lst[0]]
        prompt_2 = prompt_dict[smiles_lst[1]]
        co_g = builder.build(smiles_lst, prompt_1, prompt_2)
        graphs.append(co_g)

    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)

    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print(f'Saving graphs len(graphs)={len(valid_ids)}')
    save_graphs(cache_file_path, valid_graphs, labels={'labels': labels})

    for i, smiless in enumerate([smiless_1, smiless_2], 1):
        print(f'Extracting physicochemical features: {i}')
        fp_list = []
        des_list = []
        for smiles in smiless:
            fp, md = feat_cache.get((smiles, 3), (None, None))
            fp_list.append(fp)
            des_list.append(md.astype(np.float32))

        fp_path = f"{args.data_path}/{args.dataset}/maccs_ecfp_{i}.npz"
        des_path = f"{args.data_path}/{args.dataset}/polymer_descriptors_{i}.npz"
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

    print('Extracting ratio des')
    ratio_des = df.iloc[:, 2:5].values

    num_mapping = {0.25: 0, 0.5: 1, 0.75: 2}
    num_map = np.vectorize(lambda x: num_mapping[x])
    arch_mapping = {"alternating": 0, "block": 1, "random": 2}
    arch_map = np.vectorize(lambda x: arch_mapping[x])

    col1 = num_map(ratio_des[:, 0])
    col2 = num_map(ratio_des[:, 1])
    col3 = arch_map(ratio_des[:, 2])

    ratio_mapped = np.stack([col1, col2, col3], axis=1)
    print(f"global state: {ratio_mapped}")
    np.savez_compressed(f"{args.data_path}/{args.dataset}/ratio.npz", ratio=ratio_mapped)


if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(args)


