import multiprocessing
import pickle
from scipy import sparse as sp
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
from rdkit.Chem import MACCSkeys, AllChem
from functools import partial

import sys
sys.path.append('..')
from utils.aug import generate_multimer_smiles

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="product_smiles")
    parser.add_argument("--n_rus", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=32)
    return parser.parse_args()


def _calc_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    ec_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
    return np.array(list(map(int, list(maccs_fp + ec_fp))))


def _calc_des(smiles, calc):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(list(calc(mol).values()), dtype=np.float32)


if __name__ == '__main__':
    args = parse_args()
    smiless = pd.read_csv(f'{args.data_path}/{args.dataset}.csv').smiles.values.tolist()

    print("Preprocessing smiles")
    _generate = partial(generate_multimer_smiles, args.n_rus, replace_dummy_atoms=True)
    with multiprocessing.Pool(processes=args.n_jobs) as pool:
        smiless = list(tqdm(pool.imap(_generate, smiless, chunksize=8), total=len(smiless), ncols=100))

    print("Computing fingerprints")
    with multiprocessing.Pool(processes=args.n_jobs) as pool:
        fp_list = list(tqdm(
            pool.imap(_calc_fp, smiless, chunksize=1),
            total=len(smiless),
            ncols=100,
        ))

    fp_array = np.array(fp_list, dtype=np.float32)
    fp_sp_mat = sp.csc_matrix(fp_array)
    sp.save_npz(f"{args.data_path}/maccs_ecfp_n{args.n_rus}.npz", fp_sp_mat)

    print("Computing mordred descriptors")
    calc = Calculator(descriptors, ignore_3D=True)
    _calc_des_partial = partial(_calc_des, calc=calc)

    with multiprocessing.Pool(processes=args.n_jobs) as pool:
        des_list = list(tqdm(
            pool.imap(_calc_des_partial, smiless, chunksize=1),
            total=len(smiless),
            ncols=100,
        ))

    des_array = np.array(des_list, dtype=np.float32)
    with open("../datasets/pretrain/scaler_all.pkl", 'rb') as file:
        scaler = pickle.load(file)
    des_array = scaler.transform(des_array).astype(np.float32)
    np.savez_compressed(f"{args.data_path}/molecular_descriptors_n{args.n_rus}_norm.npz", pd=des_array)
