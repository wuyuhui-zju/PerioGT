import dgl
import torch

from utils.function import preprocess_batch_light


class CollatorFinetune(object):
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
