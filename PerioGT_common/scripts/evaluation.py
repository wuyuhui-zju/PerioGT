import sys
sys.path.append('..')

import argparse
import torch
from torch.utils.data import DataLoader

from data.finetune_dataset import PolymDataset
from data.collator_light import CollatorFinetune
from trainer.finetune_trainer import Trainer
from trainer.result_tracker import Result_Tracker
from trainer.evaluator import Evaluator
from utils.function import set_random_seed
from models.get_model_evaluation import get_model
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="light")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_prompt", type=int, default=20)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return args


def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = PolymDataset(dataset=args.dataset, split='train')
    test_dataset = PolymDataset(dataset=args.dataset, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False, collate_fn=CollatorFinetune(args.max_prompt))

    model = get_model(args)
    optimizer, lr_scheduler, loss_fn = None, None, None
    evaluator = Evaluator(args.dataset, "rmse", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    final_evaluator = Evaluator(args.dataset, "r2", train_dataset.n_tasks, mean=train_dataset.mean.numpy(), std=train_dataset.std.numpy())
    result_tracker = Result_Tracker("rmse")
    trainer = Trainer(args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator, result_tracker, device=device, label_mean=train_dataset.mean.to(device), label_std=train_dataset.std.to(device))
    best_test = trainer.eval(model, test_loader)
    best_test_r2 = trainer.eval(model, test_loader, is_test=True)

    print(f"Test RMSE: {best_test:.3f}")
    print(f"Test R2: {best_test_r2[0]:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed()
    finetune(args)
    
    

    
    
    
    
    


