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
        ecfp_path = os.path.join(root_path, f"{dataset}/maccs_ecfp.npz")
        md_path = os.path.join(root_path, f"{dataset}/polymer_descriptors.npz")
        global_state_path = os.path.join(root_path, f"{dataset}/global_state.npz")

        df = pd.read_csv(dataset_path)
        with open(split_path, 'rb') as f:
            split_idx = pickle.load(f)
        use_idxs = split_idx[SPLIT_TO_ID[split]]

        fps = torch.from_numpy(sps.load_npz(ecfp_path).todense().astype(np.float32))
        mds = torch.from_numpy(np.load(md_path)['md'].astype(np.float32))
        global_state = torch.from_numpy(np.load(global_state_path)['gs'].astype(np.float32))

        self.df, self.fps, self.mds, self.gs = df.iloc[use_idxs], fps[use_idxs], mds[use_idxs], global_state[use_idxs]

        self.smiless = self.df['smiles'].tolist()
        self.use_idxs = use_idxs

        self.task_names = ['PCE_ave']
        self.n_tasks = len(self.task_names)
        self._pre_process()
        self.mean = None
        self.std = None
        self.set_mean_and_std()
        self.d_fps = self.fps.shape[1]
        self.d_mds = self.mds.shape[1]
        self.d_gs = self.gs.shape[1]

    def _pre_process(self):
        graphs, label_dict = load_graphs(self.cache_path)
        self.graphs = []
        for i in self.use_idxs:
            self.graphs.append(graphs[i])
        self.labels = label_dict['labels'][self.use_idxs]
        self.fps, self.mds, self.gs = self.fps, self.mds, self.gs

    def __len__(self):
        return len(self.smiless)

    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx], self.gs[idx], self.labels[idx]

    def set_mean_and_std(self, mean=None, std=None):
        if mean is None:
            mean = torch.from_numpy(np.nanmean(self.labels.numpy(), axis=0))
        if std is None:
            std = torch.from_numpy(np.nanstd(self.labels.numpy(), axis=0))
        self.mean = mean
        self.std = std
