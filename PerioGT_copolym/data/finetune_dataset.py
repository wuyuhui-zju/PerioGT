import os
import pandas as pd
import numpy as np
import scipy.sparse as sps
import pickle
import torch
from torch.utils.data import Dataset
from dgl.data.utils import load_graphs


SPLIT_TO_ID = {'train': 0, 'val': 1, 'test': 2}


class PolymDataset(Dataset):
    def __init__(self, dataset, split, root_path="../datasets"):
        dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        self.cache_path = os.path.join(root_path, f"{dataset}/graphs.pkl")
        split_path = os.path.join(root_path, f"{dataset}/split.pkl")
        fp_path_1 = os.path.join(root_path, f"{dataset}/maccs_ecfp_1.npz")
        md_path_1 = os.path.join(root_path, f"{dataset}/polymer_descriptors_1.npz")
        fp_path_2 = os.path.join(root_path, f"{dataset}/maccs_ecfp_2.npz")
        md_path_2 = os.path.join(root_path, f"{dataset}/polymer_descriptors_2.npz")
        ratio_path = os.path.join(root_path, f"{dataset}/ratio.npz")
        # Load Data
        df = pd.read_csv(dataset_path)
        if split is not None:
            with open(split_path, 'rb') as f:
                split_idx = pickle.load(f)
            use_idxs = split_idx[SPLIT_TO_ID[split]]
        else:
            use_idxs = np.arange(0, len(df))
        fps_1 = torch.from_numpy(sps.load_npz(fp_path_1).todense().astype(np.float32))
        mds_1 = np.load(md_path_1)['md'].astype(np.float32)
        mds_1 = torch.from_numpy(np.where(np.isnan(mds_1), 0, mds_1))

        fps_2 = torch.from_numpy(sps.load_npz(fp_path_2).todense().astype(np.float32))
        mds_2 = np.load(md_path_2)['md'].astype(np.float32)
        mds_2 = torch.from_numpy(np.where(np.isnan(mds_2), 0, mds_2))

        ratio = np.load(ratio_path)['ratio'].astype(np.float32)
        ratio = torch.from_numpy(ratio)

        self.df, self.fps_1, self.mds_1, self.fps_2, self.mds_2 = df.iloc[use_idxs], fps_1[use_idxs], mds_1[use_idxs], fps_2[use_idxs], mds_2[use_idxs]
        self.ratio = ratio[use_idxs]

        self.smiless_1 = self.df['smiles_A'].tolist()
        self.smiless_2 = self.df['smiles_B'].tolist()
        self.use_idxs = use_idxs

        self.n_tasks = 1
        self._pre_process()
        self.mean = None
        self.std = None
        self.set_mean_and_std()
        self.d_fps = self.fps_1.shape[1]
        self.d_mds = self.mds_1.shape[1]

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
        self.fps_1, self.mds_1 = self.fps_1, self.mds_1
        self.fps_2, self.mds_2 = self.fps_2, self.mds_2

    def __len__(self):
        return len(self.smiless_1)

    def __getitem__(self, idx):
        return self.smiless_1[idx], self.smiless_2[idx], self.graphs[idx], self.fps_1[idx], self.mds_1[idx], self.fps_2[idx], self.mds_2[idx],\
            self.ratio[idx], self.labels[idx]

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std