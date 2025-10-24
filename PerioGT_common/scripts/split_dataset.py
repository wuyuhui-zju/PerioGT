import pandas as pd
import pickle
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    num_data = pd.read_csv(f'../datasets/{args.dataset}/{args.dataset}.csv').values.shape[0]
    a = np.arange(num_data)
    np.random.shuffle(a)
    split_index_1 = int(len(a) * 0.72)
    split_index_2 = int(len(a) * 0.8)

    split = []
    split.append(a[:split_index_1])
    split.append(a[split_index_1:split_index_2])
    split.append(a[split_index_2:])

    with open(f'../datasets/{args.dataset}/split.pkl', 'wb') as f:
        pickle.dump(split, f)
