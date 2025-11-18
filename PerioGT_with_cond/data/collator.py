import dgl
import torch

from utils.function import preprocess_batch_light


class CollatorFinetune(object):
    def __init__(self, len_limit=20):
        self.len_limit = len_limit

    def __call__(self, samples):
        smiles_list, graphs, fps, mds, gs, labels = map(list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list), -1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list), -1)
        gs = torch.stack(gs, dim=0).reshape(len(smiles_list), -1)

        labels = torch.stack(labels, dim=0).reshape(len(smiles_list), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(),
                                                                   batched_graph.batch_num_edges(),
                                                                   batched_graph.edata['path'][:, :])
        return smiles_list, batched_graph, fps, mds, gs, labels
