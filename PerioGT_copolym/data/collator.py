import dgl
import torch
import numpy as np


def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0],batch_num],axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate([[cs_num[i]]*batch_num_target[i] for i in range(len(cs_num)-1)], axis=-1)
    return tensor_data + torch.from_numpy(add_factors).reshape(-1,1)


class CollatorFinetune(object):
    def __init__(self, len_limit=20):
        self.len_limit = len_limit

    def __call__(self, samples):
        smiles_list_1, smiles_list_2, graphs, fps_1, mds_1, fps_2, mds_2, ratio, labels = map(list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps_1 = torch.stack(fps_1, dim=0).reshape(len(smiles_list_1), -1)
        mds_1 = torch.stack(mds_1, dim=0).reshape(len(smiles_list_1), -1)
        fps_2 = torch.stack(fps_2, dim=0).reshape(len(smiles_list_1), -1)
        mds_2 = torch.stack(mds_2, dim=0).reshape(len(smiles_list_1), -1)
        ratio = torch.stack(ratio, dim=0).reshape(len(smiles_list_1), -1)

        labels = torch.stack(labels, dim=0).reshape(len(smiles_list_1), -1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        return smiles_list_1, smiles_list_2, batched_graph, fps_1, mds_1, fps_2, mds_2, ratio, labels
