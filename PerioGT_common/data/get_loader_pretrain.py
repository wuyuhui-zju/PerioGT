import sys
sys.path.append('..')

import numpy as np
import torch
import random

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.vocab import Vocab
from data.smiles2g_light import N_BOND_TYPES, N_ATOM_TYPES
from data.pretrain_dataset import PolymDataset, PolymDatasetFF


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(args, config):
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    if args.backbone == "light":
        if not args.ff:
            from data.collator_light import CollatorPretrain
            collator = CollatorPretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2,
                                        candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'],
                                        md_disturb_rate=config['md_disturb_rate'], max_mrus=config['max_mrus'])
        else:
            from data.collator_light import CollatorPretrainFF
            collator = CollatorPretrainFF(vocab, max_length=config['path_length'], n_virtual_nodes=2,
                                        candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'],
                                        md_disturb_rate=config['md_disturb_rate'], max_mrus=config['max_mrus'])
    elif args.backbone == "graphgps":
        if not args.ff:
            from data.collator_graphgps import CollatorPretrain
            collator = CollatorPretrain(vocab, n_virtual_nodes=2, candi_rate=config['candi_rate'],
                                        fp_disturb_rate=config['fp_disturb_rate'],
                                        md_disturb_rate=config['md_disturb_rate'])
        else:
            from data.collator_graphgps import CollatorPretrainFF
            collator = CollatorPretrainFF(vocab, n_virtual_nodes=2, candi_rate=config['candi_rate'],
                                        fp_disturb_rate=config['fp_disturb_rate'],
                                        md_disturb_rate=config['md_disturb_rate'])
    else:
        raise ValueError("Backbone must in ['light', 'graphgps']")

    if not args.ff:
        train_dataset = PolymDataset(root_path=args.data_path)
    else:
        train_dataset = PolymDatasetFF(root_path=args.data_path)

    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset),
                              batch_size=config['batch_size'] // args.n_devices, num_workers=args.n_threads,
                              worker_init_fn=seed_worker, drop_last=True, collate_fn=collator)

    return train_dataset, train_loader
