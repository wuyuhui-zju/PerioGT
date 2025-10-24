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
    def __init__(self, dataset, split, root_path="../datasets", mode="finetune"):
        dataset_path = os.path.join(root_path, f"{dataset}/{dataset}.csv")
        if mode == 'finetune':
            self.cache_path = os.path.join(root_path, f"{dataset}/graphs.pkl")
        elif mode == 'feature':
            self.cache_path = os.path.join(root_path, f"{dataset}/graphs_feat.pkl")
        split_path = os.path.join(root_path, f"{dataset}/split.pkl")
        ecfp_path = os.path.join(root_path, f"{dataset}/maccs_ecfp.npz")
        md_path = os.path.join(root_path, f"{dataset}/polymer_descriptors.npz")
        # Load Data
        df = pd.read_csv(dataset_path)
        if split is not None:
            with open(split_path, 'rb') as f:
                split_idx = pickle.load(f)
            use_idxs = split_idx[SPLIT_TO_ID[split]]
        else:
            use_idxs = np.arange(0, len(df))

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = np.load(md_path)['md'].astype(np.float32)
        mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
        self.df, self.fps, self.mds = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs]
        self.smiless = self.df['smiles'].tolist()
        self.use_idxs = use_idxs

        # Dataset Setting
        self.task_names = self.df.columns.drop(['smiles']).tolist()
        self.n_tasks = len(self.task_names)
        self._pre_process()
        self.mean = None
        self.std = None
        self.set_mean_and_std()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]

    def _pre_process(self):
        if not os.path.exists(self.cache_path):
            print(f"{self.cache_path} not exists, please run preprocess.py")
        else:
            graphs, label_dict = load_graphs(self.cache_path)
            self.graphs = []
            for i in self.use_idxs:
                self.graphs.append(graphs[i])
            self.labels = label_dict['labels'][self.use_idxs]
        self.fps, self.mds = self.fps, self.mds

    def __len__(self):
        return len(self.smiless)
    
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx], self.labels[idx]

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std