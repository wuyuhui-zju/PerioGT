import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm

import sys
sys.path.append("..")
from data.collator_light import CollatorExtract
from data.finetune_dataset import PolymDataset
from models.get_model_pretrain import get_model
from utils.function import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="light")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    return args


def extract_features(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    collator = CollatorExtract()
    dataset = PolymDataset(dataset=args.dataset, split=None, root_path=args.data_path, mode="feature")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False, collate_fn=collator)

    model = get_model(args, device)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model_path).items()})
    model.eval()

    feat_list = []
    with torch.no_grad():
        for batched_data in tqdm(loader, total=len(loader)):
            (_, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(device)
            md = md.to(device)
            g = g.to(device)
            feat = model.generate_features(g, ecfp, md)
            feat_list.extend(feat.detach().cpu().numpy().tolist())

    print("Saving features")
    np.savez_compressed(f"{args.data_path}/{args.dataset}/features.npz", fps=np.array(feat_list))


if __name__ == '__main__':
    set_random_seed(22)
    args = parse_args()
    extract_features(args)
