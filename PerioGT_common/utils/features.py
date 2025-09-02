from mordred import Calculator, descriptors
import numpy as np
from rdkit.Chem import MACCSkeys, AllChem
from rdkit import Chem

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from utils.aug import generate_oligomer_smiles

_GLOBAL_CALC = Calculator(descriptors, ignore_3D=True)


def _fp_md_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    ec_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
    fp = list(map(int, list(maccs_fp + ec_fp)))

    # Mordred
    des = np.array(list(_GLOBAL_CALC(mol).values()), dtype=np.float32)
    des = np.where(np.isnan(des), 0, des)
    des = np.where(des > 1e12, 1e12, des)
    return smiles, fp, des


def _gen_oligomer_smiles(base: str, units: int):
    try:
        return generate_oligomer_smiles(num_repeat_units=units, smiles=base)
    except Exception:
        return None


def precompute_features(base_smiles_list, units=(3,6,9), workers=None):
    workers = workers or max(1, cpu_count() - 1)

    tasks = [(s, u) for s in base_smiles_list for u in units]
    with Pool(processes=workers) as pool:
        gen_smiles = pool.starmap(_gen_oligomer_smiles, tasks, chunksize=64)

    oligos = [(base, u, sm) for (base, u), sm in zip(tasks, gen_smiles)]

    oligo_smiles_map = {}
    uniq_smiles_set = set()
    for base, u, sm in oligos:
        if sm is not None:
            oligo_smiles_map[(base, u)] = sm
            uniq_smiles_set.add(sm)

    # multiprocessing
    uniq_smiles = list(uniq_smiles_set)
    with Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_fp_md_from_smiles, uniq_smiles, chunksize=1),
            total=len(uniq_smiles),
            ncols=100,
        ))

    fpmd_map = {sm: (fp, md) for sm, fp, md in results}

    feat_cache = {}
    for k, sm in oligo_smiles_map.items():
        feat_cache[k] = fpmd_map.get(sm, (None, None))

    return feat_cache
