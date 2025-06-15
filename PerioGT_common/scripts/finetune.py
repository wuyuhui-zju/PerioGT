import sys
sys.path.append('..')

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import warnings
import numpy as np
import random

from data.finetune_dataset import PolymDataset
from data.collator_light import CollatorFinetune
from models.get_model_finetune import get_model
from trainer.scheduler import PolynomialDecayLR
from trainer.finetune_trainer import Trainer
from trainer.result_tracker import Result_Tracker
from trainer.evaluator import Evaluator
from utils.function import set_random_seed
warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--n_epochs", type=int, default=50)

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="light")
    parser.add_argument("--mode", type=str, default="finetune")
    parser.add_argument("--model_path", type=str, default="../checkpoints/pretrained/light/base.pth")
    parser.add_argument("--max_prompt", type=int, default=20)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save_suffix", type=str)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    return args


def finetune(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    g = torch.Generator()
    g.manual_seed(args.seed)
    collator = CollatorFinetune(args.max_prompt)
    train_dataset = PolymDataset(dataset=args.dataset, split='train')
    val_dataset = PolymDataset(dataset=args.dataset, split='val')
    test_dataset = PolymDataset(dataset=args.dataset, split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g, drop_last=False, collate_fn=collator)
    model = get_model(args)
    node_attn_params = list(map(id, model.node_attn.parameters()))
    if args.mode == "freeze":
        base_params = filter(lambda p: id(p) not in node_attn_params and p.requires_grad, model.parameters())
    elif args.mode == "finetune":
        base_params = filter(lambda p: id(p) not in node_attn_params, model.parameters())
    param_groups = [
        {'params': base_params},
        {'params': model.node_attn.parameters(), 'lr': args.lr * 5}
    ]
    optimizer = Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs * len(train_dataset) // 32 // 10,
                                     tot_updates=args.n_epochs * len(train_dataset) // 32, lr=args.lr, end_lr=1e-9, power=1)

    loss_fn = MSELoss(reduction='none')
    evaluator = Evaluator(args.dataset, "rmse", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator = Evaluator(args.dataset, "r2", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    result_tracker = Result_Tracker("rmse")

    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator, result_tracker, device,
                      label_mean=train_dataset.mean.to(device), label_std=train_dataset.std.to(device))
    best_train, best_val, best_test, test_final = trainer.fit(model, train_loader, val_loader, test_loader)
    print(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")
    print(f"test r2: {test_final[0]:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    finetune(args)
    
    

    
    
    
    
    


