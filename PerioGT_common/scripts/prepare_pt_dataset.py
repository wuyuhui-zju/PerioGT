import multiprocessing
import gc
from time import time
from scipy import sparse as sp
import argparse
from tqdm import tqdm
from rdkit import Chem
from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys, AllChem
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args


def doit(args, ind, sub_df):
    product_list = sub_df.product_smiles.values.tolist()

    print(f"calculating mordred descriptors: {ind}")
    des_list = []
    for smiles in tqdm(product_list, total=len(product_list), desc=f"Processing: {ind}", ncols=100):
        calc = Calculator(descriptors, ignore_3D=True)
        mol = Chem.MolFromSmiles(smiles)
        des = np.array(list(calc(mol).values()), dtype=np.float32)
        des_list.append(des)

    des = np.array(des_list)
    gc.collect()
    np.savez_compressed(f"{args.data_path}/polymer_descriptors_{ind}.npz", pd=des)

    print(f"calculating fingerprint: {ind}")
    fp_list = []
    for smiles in tqdm(product_list, total=len(product_list), desc=f"Processing: {ind}", ncols=100):
        mol = Chem.MolFromSmiles(smiles)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        ec_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        fp_list.append(list(map(int, list(maccs_fp + ec_fp))))

    fp_list = np.array(fp_list, dtype=np.float32)
    gc.collect()
    fp_sp_mat = sp.csc_matrix(fp_list)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/maccs_ecfp_{ind}.npz", fp_sp_mat)


if __name__ == '__main__':
    args = parse_args()
    t1 = time()
    pros = []

    df = pd.read_csv(f'{args.data_path}/product_smiles.csv')
    sub_dfs = np.array_split(df, args.n_jobs)
    for i, df in enumerate(sub_dfs):
        process = multiprocessing.Process(target=doit, args=(args, i, df))
        pros.append(process)
        process.start()

    for process in pros:
        process.join()

    t2 = time()
    print(t2 - t1)
