import sys
sys.path.append('..')

import numpy as np
import torch
import random

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.vocab import Vocab
from data.pretrain_dataset import PolymDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset(args, config):
    vocab = Vocab()
    if args.backbone == "light":
        from data.collator_light import CollatorPretrain
        collator = CollatorPretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2,
                                    candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'],
                                    md_disturb_rate=config['md_disturb_rate'], max_mrus=config['max_mrus'])

    elif args.backbone == "graphgps":
        from data.collator_graphgps import CollatorPretrain
        collator = CollatorPretrain(vocab, pos_enc_size=config['pos_enc_size'], n_virtual_nodes=2, candi_rate=config['candi_rate'],
                                    fp_disturb_rate=config['fp_disturb_rate'],
                                    md_disturb_rate=config['md_disturb_rate'])

    else:
        raise ValueError("Backbone must in ['light', 'graphgps']")


    train_dataset = PolymDataset(root_path=args.data_path)
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset),
                              batch_size=config['batch_size'] // args.n_devices, num_workers=args.n_threads,
                              worker_init_fn=seed_worker, drop_last=True, collate_fn=collator)

    return train_dataset, train_loader
