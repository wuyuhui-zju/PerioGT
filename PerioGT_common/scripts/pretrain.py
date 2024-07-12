import sys
sys.path.append('..')

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import random
from data.vocab import Vocab
from data.smiles2g_light import N_BOND_TYPES, N_ATOM_TYPES
from data.pretrain_dataset import PolymDataset

from trainer.scheduler import PolynomialDecayLR
from trainer.pretrain_trainer import Trainer
from trainer.evaluator import Evaluator
from trainer.result_tracker import Result_Tracker
from utils.loss import DistributedNCELoss
from utils.function import set_random_seed, load_config
from models.get_model_pretrain import get_model
import warnings


warnings.filterwarnings("ignore")
local_rank = int(os.environ['LOCAL_RANK'])


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_steps", type=int, required=True)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--backbone", type=str, default="light")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_devices", type=int, default=1)
    args = parser.parse_args()
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    val_results, test_results, train_results = [], [], []
    
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    if args.backbone == "light":
        from data.collator_light import CollatorPretrain
        collator = CollatorPretrain(vocab, max_length=config['path_length'], n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'], max_mrus=config['max_mrus'])
    elif args.backbone == "graphgps":
        from data.collator_graphgps import CollatorPretrain
        collator = CollatorPretrain(vocab, n_virtual_nodes=2, candi_rate=config['candi_rate'], fp_disturb_rate=config['fp_disturb_rate'], md_disturb_rate=config['md_disturb_rate'])

    train_dataset = PolymDataset(root_path=args.data_path)
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset), batch_size=config['batch_size']// args.n_devices, num_workers=args.n_threads, worker_init_fn=seed_worker, drop_last=True, collate_fn=collator)

    model = get_model(args, device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=20000, tot_updates=200000, lr=config['lr'], end_lr=1e-9, power=1)
    reg_loss_fn = MSELoss(reduction='none')
    clf_loss_fn = BCEWithLogitsLoss(weight=train_dataset._task_pos_weights.to(device), reduction='none')
    sl_loss_fn = CrossEntropyLoss(reduction='none')
    nce_loss_fn = DistributedNCELoss()
    reg_metric, clf_metric = "r2", "rocauc_resp"
    reg_evaluator = Evaluator("pretrain", reg_metric, train_dataset.d_mds)
    clf_evaluator = Evaluator("pretrain", clf_metric, train_dataset.d_fps)
    result_tracker = Result_Tracker(reg_metric)

    trainer = Trainer(args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, nce_loss_fn, reg_evaluator, clf_evaluator, result_tracker, device=device, local_rank=local_rank)
    trainer.fit(model, train_loader)


    
    

