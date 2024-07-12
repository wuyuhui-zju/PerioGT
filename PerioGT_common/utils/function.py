import scipy.sparse as sparse
import dgl.backend as F
import os
import random
import torch
import dgl
import numpy as np
import yaml


def load_config(args, file_path="../config.yaml"):
    with open(file_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return config_dict[args.backbone][args.config]


def set_random_seed(seed=22, n_threads=16):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed)


def random_walk_pe(g, k, eweight_name=None):
    r"""Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This function computes the random walk positional encodings as landing probabilities
    from 1-step to k-step, starting from each node to itself.

    Parameters
    ----------
    g : DGLGraph
        The input graph. Must be homogeneous.
    k : int
        The number of random walk steps. The paper found the best value to be 16 and 20
        for two experiments.
    eweight_name : str, optional
        The name to retrieve the edge weights. Default: None, not using the edge weights.

    Returns
    -------
    Tensor
        The random walk positional encodings of shape :math:`(N, k)`, where :math:`N` is the
        number of nodes in the input graph.
    """
    N = g.num_nodes() # number of nodes
    M = g.num_edges() # number of edges
    A = g.adj(scipy_fmt='csr') # adjacency matrix
    if eweight_name is not None:
        # add edge weights if required
        W = sparse.csr_matrix(
            (g.edata[eweight_name].squeeze(), g.find_edges(list(range(M)))),
            shape = (N, N)
        )
        A = A.multiply(W)
    RW = np.array(A / (A.sum(1) + 1e-30)) # 1-step transition probability

    # Iterate for k steps
    PE = [F.astype(F.tensor(RW.diagonal()), F.float32)]
    RW_power = RW
    for _ in range(k-1):
        RW_power = RW_power @ RW
        PE.append(F.astype(F.tensor(RW_power.diagonal()), F.float32))
    PE = F.stack(PE,dim=-1)

    return PE


if __name__ == "__main__":
    g = dgl.graph(([0, 1, 1], [1, 1, 0]))
    print(random_walk_pe(g, 2))