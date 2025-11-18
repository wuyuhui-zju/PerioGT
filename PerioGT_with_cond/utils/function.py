import scipy.sparse as sparse
import dgl.backend as F
import os
import random
import torch
import dgl
import numpy as np
import yaml
import multiprocessing
import cloudpickle


def load_config(args, file_path="../config.yaml"):
    with open(file_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return config_dict[args.config]


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


def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0],batch_num],axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate([[cs_num[i]]*batch_num_target[i] for i in range(len(cs_num)-1)], axis=-1)
    return tensor_data + torch.from_numpy(add_factors).reshape(-1,1)


class SafeSubstructureMatcher:
    def __init__(self, timeout=3):
        self.timeout = timeout
        self._start_worker()

    def _start_worker(self):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._worker_loop)
        self.process.daemon = True
        self.process.start()

    def _restart_worker(self):
        self.process.terminate()
        self.process.join()
        self._start_worker()

    def _worker_loop(self):
        while True:
            try:
                mol_data, patt_data = self.task_queue.get()
                mol = cloudpickle.loads(mol_data)
                patt = cloudpickle.loads(patt_data)
                if mol is not None and patt is not None:
                    match = mol.GetSubstructMatch(patt)
                    self.result_queue.put(match)
                else:
                    self.result_queue.put(())
            except Exception as e:
                self.result_queue.put(())

    def match(self, mol, patt):
        mol_data = cloudpickle.dumps(mol)
        patt_data = cloudpickle.dumps(patt)

        self.task_queue.put((mol_data, patt_data))
        try:
            return self.result_queue.get(timeout=self.timeout)
        except multiprocessing.queues.Empty:
            self._restart_worker()
            return ()

    def shutdown(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


if __name__ == "__main__":
    g = dgl.graph(([0, 1, 1], [1, 1, 0]))
    print(random_walk_pe(g, 2))
