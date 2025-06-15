import sys
sys.path.append('..')

import argparse
import torch
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
import os

from trainer.scheduler import PolynomialDecayLR
from trainer.pretrain_trainer import Trainer
from trainer.evaluator import Evaluator
from trainer.result_tracker import Result_Tracker
from utils.loss import DistributedNCELoss
from utils.function import set_random_seed, load_config
from models.get_model_pretrain import get_model
from data.get_loader_pretrain import get_dataset
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
    parser.add_argument("--ff", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    val_results, test_results, train_results = [], [], []

    train_dataset, train_loader = get_dataset(args, config)
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


    
    

