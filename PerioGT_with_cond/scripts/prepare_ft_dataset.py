import sys
sys.path.append('..')

import argparse
import os
import pickle
import pandas as pd
from dgl import save_graphs
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
import numpy as np
from rdkit.Chem import MACCSkeys, AllChem
from rdkit import Chem
import dgl.backend as F
import torch
from tqdm import tqdm
from scipy import sparse as sp
import warnings

from data.vocab import Vocab
from data.smiles2g import smiles_to_graph_with_prompt
from utils.aug import generate_oligomer_smiles
from models.light import LiGhTPredictor as LiGhT
from utils.function import load_config

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--model_path", type=str, default="../checkpoints/pretrained/light/base.pth")
    parser.add_argument("--data_path", type=str, default='../datasets/')

    args = parser.parse_args()
    return args


def get_model(args):
    config = load_config(args)
    vocab = Vocab()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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

    model = model.to(torch.device(device))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.model_path).items()})
    return model


def prepare_dataset(args):
    with open("../datasets/pretrain/scaler_all.pkl", 'rb') as file:
        scaler = pickle.load(file)

    df = pd.read_csv(f"{args.data_path}/{args.dataset}/{args.dataset}.csv")
    cache_file_path = f"{args.data_path}/{args.dataset}/graphs.pkl"
    smiless = df.smiles.values.tolist()

    monomer_list = []
    product_list = []
    for smiles in smiless:
        try:
            product_smiles = generate_oligomer_smiles(num_repeat_units=3, smiles=smiles)
            monomer_list.append([smiles])
            product_list.append(product_smiles)
        except:
            print(smiles)
            continue
    task_names = ['PCE_ave']
    print('constructing graphs')
    model = get_model(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    graphs = []
    for smiles_lst in tqdm(monomer_list, total=len(monomer_list), ncols=100):
        g = smiles_to_graph_with_prompt(smiles_lst, model, scaler, device, n_knowledge_nodes=2, n_global_nodes=1)
        graphs.append(g)

    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)

    _label_values = df[task_names].values
    labels = F.zerocopy_from_numpy(
        _label_values.astype(np.float32))[valid_ids]
    print(f'saving graphs len(graphs)={len(valid_ids)}')
    save_graphs(cache_file_path, valid_graphs,
                labels={'labels': labels})

    print('extracting fingerprints')
    fp_list = []
    for smiles in product_list:
        mol = Chem.MolFromSmiles(smiles)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        ec_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        fp_list.append(list(map(int, list(maccs_fp + ec_fp))))
    fp_list = np.array(fp_list, dtype=np.float32)
    print(f"fp shape: {fp_list.shape}")
    fp_sp_mat = sp.csc_matrix(fp_list)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/{args.dataset}/maccs_ecfp.npz", fp_sp_mat)

    print('extracting polymer descriptors')
    des_list = []
    for smiles in tqdm(product_list, total=len(product_list)):
        calc = Calculator(descriptors, ignore_3D=True)
        mol = Chem.MolFromSmiles(smiles)
        des = np.array(list(calc(mol).values()), dtype=np.float32)
        des_list.append(des)
    des = np.array(des_list)
    des = np.where(np.isnan(des), 0, des)
    des = np.where(des > 10 ** 12, 10 ** 12, des)
    des_norm = scaler.transform(des)

    print(f"des shape: {des.shape}")
    np.savez_compressed(f"{args.data_path}/{args.dataset}/polymer_descriptors.npz", md=des_norm)

    print('extracting global state')
    global_state = df.iloc[:, 1:-1].values
    print(f"global state shape: {global_state.shape}")
    np.savez_compressed(f"{args.data_path}/{args.dataset}/global_state.npz", gs=global_state)


if __name__ == '__main__':
    args = parse_args()
    prepare_dataset(args)


