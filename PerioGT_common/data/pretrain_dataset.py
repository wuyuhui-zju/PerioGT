from torch.utils.data import Dataset
import os
import numpy as np
import scipy.sparse as sps
import pandas as pd
import torch
import dgl.backend as F


class PolymDataset(Dataset):
    def __init__(self, root_path):
        smiles_path = os.path.join(root_path, "product_smiles.csv")
        fp_path_1 = os.path.join(root_path, "maccs_ecfp_n3.npz")
        fp_path_2 = os.path.join(root_path, "maccs_ecfp_n6.npz")
        fp_path_3 = os.path.join(root_path, "maccs_ecfp_n9.npz")
        md_path_1 = os.path.join(root_path, "molecular_descriptors_n3_norm.npz")
        md_path_2 = os.path.join(root_path, "molecular_descriptors_n6_norm.npz")
        md_path_3 = os.path.join(root_path, "molecular_descriptors_n9_norm.npz")
        self.smiles_list = pd.read_csv(smiles_path).smiles.values.tolist()
        self.fps_1 = torch.from_numpy(sps.load_npz(fp_path_1).todense().astype(np.float32))
        self.fps_2 = torch.from_numpy(sps.load_npz(fp_path_2).todense().astype(np.float32))
        self.fps_3 = torch.from_numpy(sps.load_npz(fp_path_3).todense().astype(np.float32))

        self.mds_1 = torch.from_numpy(np.load(md_path_1)['md'].astype(np.float32))
        self.mds_2 = torch.from_numpy(np.load(md_path_2)['md'].astype(np.float32))
        self.mds_3 = torch.from_numpy(np.load(md_path_3)['md'].astype(np.float32))
        self.d_fps = self.fps_1.shape[1]
        self.d_mds = self.mds_1.shape[1]

        self._task_pos_weights = self.task_pos_weights()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx], self.fps_1[idx], self.fps_2[idx], self.fps_3[idx], \
            self.mds_1[idx], self.mds_2[idx], self.mds_3[idx]

    def task_pos_weights(self):
        task_pos_weights = torch.ones(self.fps_1.shape[1])
        num_pos = torch.sum(torch.nan_to_num(self.fps_1, nan=0), axis=0)
        masks = F.zerocopy_from_numpy(
            (~np.isnan(self.fps_1.numpy())).astype(np.float32))
        num_indices = torch.sum(masks, axis=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]
        return task_pos_weights
