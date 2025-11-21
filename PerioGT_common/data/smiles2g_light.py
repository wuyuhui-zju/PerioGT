import torch
import dgl
import numpy as np
import random
from rdkit import Chem
from dgllife.utils.featurizers import ConcatFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, \
    bond_stereo_one_hot, atomic_number_one_hot, atom_degree_one_hot, atom_formal_charge, \
    atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_is_aromatic, atom_total_num_H_one_hot, \
    atom_is_chiral_center, atom_chirality_type_one_hot, atom_mass
from functools import partial
from itertools import permutations
import networkx as nx

from utils.aug import generate_multimer_smiles, periodicity_augment_traverse
from utils.function import preprocess_batch_light, SafeSubstructureMatcher
from data.vocab import Vocab
from data.constants import INF, VIRTUAL_ATOM_INDICATOR, VIRTUAL_ATOM_FEATURE_PLACEHOLDER,\
    VIRTUAL_BOND_FEATURE_PLACEHOLDER, VIRTUAL_PATH_INDICATOR, N_ATOM_TYPES, N_BOND_TYPES
from models.light import LiGhTPredictor as LiGhT
from utils.function import load_config


class PolyGraphBuilderPretrain:
    def __init__(self, vocab, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.atom_featurizer = ConcatFeaturizer([  # 138
            partial(atomic_number_one_hot, allowable_set=list(range(0, 101)), encode_unknown=True),  # 102
            partial(atom_degree_one_hot, encode_unknown=True),  # 12
            atom_formal_charge,  # 1
            partial(atom_num_radical_electrons_one_hot, encode_unknown=True),  # 6
            partial(atom_hybridization_one_hot, encode_unknown=True),  # 6
            atom_is_aromatic,  # 1
            partial(atom_total_num_H_one_hot, encode_unknown=True),  # 6
            atom_is_chiral_center,  # 1
            atom_chirality_type_one_hot,  # 2
            atom_mass,  # 1
        ])
        self.bond_featurizer = ConcatFeaturizer([  # 14
            partial(bond_type_one_hot, encode_unknown=True),  # 5
            bond_is_conjugated,  # 1
            bond_is_in_ring,  # 1
            partial(bond_stereo_one_hot, encode_unknown=True)  # 7
        ])
        self.vocab = vocab
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

    def build(self, smiles: str):
        d_atom_feats = 138
        d_bond_feats = 14
        # Canonicalize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
        # Featurize Atoms
        n_atoms = mol.GetNumAtoms()
        atom_features = []

        for atom_id in range(n_atoms):
            atom = mol.GetAtomWithIdx(atom_id)
            atom_features.append(self.atom_featurizer(atom))
        atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan
        # Construct and Featurize Triplet Nodes
        ## bonded atoms
        triplet_labels = []
        virtual_atom_and_virtual_node_labels = []

        atom_pairs_features_in_triplets = []
        bond_features_in_triplets = []

        bonded_atoms = set()
        triplet_id = 0
        for bond in mol.GetBonds():
            begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
            atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
            bond_feature = self.bond_featurizer(bond)
            bond_features_in_triplets.append(bond_feature)
            bonded_atoms.add(begin_atom_id)
            bonded_atoms.add(end_atom_id)
            triplet_labels.append(self.vocab.index(atom_features[begin_atom_id][:N_ATOM_TYPES].index(1),
                                              atom_features[end_atom_id][:N_ATOM_TYPES].index(1),
                                              bond_feature[:N_BOND_TYPES].index(1)))
            virtual_atom_and_virtual_node_labels.append(0)
            atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[
                end_atom_id, begin_atom_id] = triplet_id
            triplet_id += 1
        ## unbonded atoms
        for atom_id in range(n_atoms):
            if atom_id not in bonded_atoms:
                atom_pairs_features_in_triplets.append(
                    [atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
                bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
                triplet_labels.append(self.vocab.index(atom_features[atom_id][:N_ATOM_TYPES].index(1), 999, 999))
                virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)
        # Construct and Featurize Paths between Triplets
        ## line graph paths
        edges = []
        paths = []
        line_graph_path_labels = []
        mol_graph_path_labels = []
        virtual_path_labels = []
        self_loop_labels = []
        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)
                new_paths = [[new_edge[0]] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [new_edge[1]] for new_edge in new_edges]
                paths.extend(new_paths)
                n_new_edges = len(new_edges)
                line_graph_path_labels.extend([1] * n_new_edges)
                mol_graph_path_labels.extend([0] * n_new_edges)
                virtual_path_labels.extend([0] * n_new_edges)
                self_loop_labels.extend([0] * n_new_edges)
        # # molecule graph paths
        adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
        nx_g = nx.from_numpy_array(adj_matrix)
        paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, self.max_length + 1))
        for i in paths_dict.keys():
            for j in paths_dict[i]:
                path = paths_dict[i][j]
                path_length = len(path)
                if 3 < path_length <= self.max_length + 1:
                    triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]] for pi in range(len(path) - 1)]
                    path_start_triplet_id = triplet_ids[0]
                    path_end_triplet_id = triplet_ids[-1]
                    triplet_path = triplet_ids[1:-1]
                    triplet_path = [path_start_triplet_id] + triplet_path + [VIRTUAL_PATH_INDICATOR] * (self.max_length - len(triplet_path) - 2) + [path_end_triplet_id]
                    paths.append(triplet_path)
                    edges.append([path_start_triplet_id, path_end_triplet_id])
                    line_graph_path_labels.append(0)
                    mol_graph_path_labels.append(1)
                    virtual_path_labels.append(0)
                    self_loop_labels.append(0)

        for n in range(self.n_virtual_nodes):
            for i in range(len(atom_pairs_features_in_triplets) - n):
                edges.append([len(atom_pairs_features_in_triplets), i])
                edges.append([i, len(atom_pairs_features_in_triplets)])
                paths.append([len(atom_pairs_features_in_triplets)] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [len(atom_pairs_features_in_triplets)])
                line_graph_path_labels.extend([0, 0])
                mol_graph_path_labels.extend([0, 0])
                virtual_path_labels.extend([n + 1, n + 1])
                self_loop_labels.extend([0, 0])
            atom_pairs_features_in_triplets.append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
            triplet_labels.append(self.vocab.index(999, 999, 999))
            virtual_atom_and_virtual_node_labels.append(n + 1)

        if self.add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i, i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(0)
                virtual_path_labels.append(0)
                self_loop_labels.append(1)

        edges = np.array(edges, dtype=np.int64)
        data = (edges[:, 0], edges[:, 1])
        g = dgl.graph(data)
        g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
        g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
        g.ndata['label'] = torch.LongTensor(triplet_labels)
        g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
        g.edata['path'] = torch.LongTensor(paths)
        g.edata['lgp'] = torch.BoolTensor(line_graph_path_labels)
        g.edata['mgp'] = torch.BoolTensor(mol_graph_path_labels)
        g.edata['vp'] = torch.BoolTensor(virtual_path_labels)
        g.edata['sl'] = torch.BoolTensor(self_loop_labels)
        return g


class PolyGraphBuilderFinetune:
    def __init__(self, args, scaler, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.args = args
        self.scaler = scaler
        self.atom_featurizer = ConcatFeaturizer([  # 138
            partial(atomic_number_one_hot, allowable_set=list(range(0, 101)), encode_unknown=True),  # 102
            partial(atom_degree_one_hot, encode_unknown=True),  # 12
            atom_formal_charge,  # 1
            partial(atom_num_radical_electrons_one_hot, encode_unknown=True),  # 6
            partial(atom_hybridization_one_hot, encode_unknown=True),  # 6
            atom_is_aromatic,  # 1
            partial(atom_total_num_H_one_hot, encode_unknown=True),  # 6
            atom_is_chiral_center,  # 1
            atom_chirality_type_one_hot,  # 2
            atom_mass,  # 1
        ])
        self.bond_featurizer = ConcatFeaturizer([  # 14
            partial(bond_type_one_hot, encode_unknown=True),  # 5
            bond_is_conjugated,  # 1
            bond_is_in_ring,  # 1
            partial(bond_stereo_one_hot, encode_unknown=True)  # 7
        ])
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
        self.use_prompt = self.args.use_prompt
        if self.use_prompt:
            self.device = self.args.device
            self.model = self._get_model()
            self.max_prompt = self.args.max_prompt
            self.matcher = SafeSubstructureMatcher()

    def build(self, smiles: str, feat_cache=None):
        if self.use_prompt:
            return self._build_with_prompt(smiles, feat_cache=feat_cache)
        else:
            return self._build_without_prompt(smiles)

    def _build_with_prompt(self, base_smiles: str, feat_cache: dict):
        assert feat_cache is not None, "feat_cache is required when use_prompt=True"

        graphs, smiless, smiless_n = self._generate_augmented_graphs(base_smiles)
        base_graph = graphs[0]

        fp_list = []
        des_list = []
        for n in smiless_n:
            u = self._map_n_to_units(n)
            fp, md = feat_cache.get((base_smiles, u), (None, None))
            fp_list.append(fp)
            des_list.append(md)

        fp_list = np.array(fp_list, dtype=np.float32)
        des = np.array(des_list, dtype=np.float32)
        des_norm = self.scaler.transform(des).astype(np.float32)

        graphs = dgl.batch(graphs)
        graphs.edata['path'][:, :] = preprocess_batch_light(graphs.batch_num_nodes(), graphs.batch_num_edges(), graphs.edata['path'][:, :])
        graphs = graphs.to(self.device)
        fps = torch.from_numpy(fp_list).to(self.device)
        mds = torch.from_numpy(des_norm).to(self.device)

        node_embs, bids = self.model.generate_node_emb(graphs, fps, mds)
        bids = bids.cpu().numpy().tolist()
        BID_to_PROMPT_DICT = {bid: [] for bid in bids}
        for i, bid in enumerate(bids):
            prompt = node_embs[i]
            BID_to_PROMPT_DICT[bid].append(prompt.detach().cpu())

        for k, v in BID_to_PROMPT_DICT.items():
            BID_to_PROMPT_DICT[k] = torch.stack(v)

        feats = torch.zeros([base_graph.number_of_nodes(), self.max_prompt, BID_to_PROMPT_DICT[-1].size()[1]])
        for nid in range(base_graph.number_of_nodes()):
            bid = base_graph.ndata['bid'][nid].item()
            feat = BID_to_PROMPT_DICT[bid]
            L = min(feat.size(0), self.max_prompt)
            feats[nid][:L, :] = feat[:L]

        base_graph.ndata["prompt"] = feats
        return base_graph

    def _build_without_prompt(self, smiles: str):
        d_atom_feats = 138
        d_bond_feats = 14
        # Canonicalize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
        # Featurize Atoms
        n_atoms = mol.GetNumAtoms()
        atom_features = []

        for atom_id in range(n_atoms):
            atom = mol.GetAtomWithIdx(atom_id)
            atom_features.append(self.atom_featurizer(atom))
        atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan
        # Construct and Featurize Triplet Nodes
        ## bonded atoms
        virtual_atom_and_virtual_node_labels = []

        atom_pairs_features_in_triplets = []
        bond_features_in_triplets = []

        bonded_atoms = set()
        triplet_id = 0
        for bond in mol.GetBonds():
            begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
            atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
            bond_feature = self.bond_featurizer(bond)
            bond_features_in_triplets.append(bond_feature)
            bonded_atoms.add(begin_atom_id)
            bonded_atoms.add(end_atom_id)
            virtual_atom_and_virtual_node_labels.append(0)
            atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[
                end_atom_id, begin_atom_id] = triplet_id
            triplet_id += 1

        ## unbonded atoms
        for atom_id in range(n_atoms):
            if atom_id not in bonded_atoms:
                atom_pairs_features_in_triplets.append(
                    [atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
                bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
                virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)
        # Construct and Featurize Paths between Triplets
        ## line graph paths
        edges = []
        paths = []
        line_graph_path_labels = []
        mol_graph_path_labels = []
        virtual_path_labels = []
        self_loop_labels = []
        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)
                new_paths = [[new_edge[0]] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [new_edge[1]] for new_edge in new_edges]
                paths.extend(new_paths)
                n_new_edges = len(new_edges)
                line_graph_path_labels.extend([1] * n_new_edges)
                mol_graph_path_labels.extend([0] * n_new_edges)
                virtual_path_labels.extend([0] * n_new_edges)
                self_loop_labels.extend([0] * n_new_edges)
        # # molecule graph paths
        adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
        nx_g = nx.from_numpy_array(adj_matrix)
        paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, self.max_length + 1))
        for i in paths_dict.keys():
            for j in paths_dict[i]:
                path = paths_dict[i][j]
                path_length = len(path)
                if 3 < path_length <= self.max_length + 1:
                    triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]] for pi in range(len(path) - 1)]
                    path_start_triplet_id = triplet_ids[0]
                    path_end_triplet_id = triplet_ids[-1]
                    triplet_path = triplet_ids[1:-1]
                    # assert [path_start_triplet_id,path_end_triplet_id] not in edges
                    triplet_path = [path_start_triplet_id] + triplet_path + [VIRTUAL_PATH_INDICATOR] * (
                            self.max_length - len(triplet_path) - 2) + [path_end_triplet_id]
                    paths.append(triplet_path)
                    edges.append([path_start_triplet_id, path_end_triplet_id])
                    line_graph_path_labels.append(0)
                    mol_graph_path_labels.append(1)
                    virtual_path_labels.append(0)
                    self_loop_labels.append(0)

        for n in range(self.n_virtual_nodes):
            for i in range(len(atom_pairs_features_in_triplets) - n):
                edges.append([len(atom_pairs_features_in_triplets), i])
                edges.append([i, len(atom_pairs_features_in_triplets)])
                paths.append([len(atom_pairs_features_in_triplets)] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [len(atom_pairs_features_in_triplets)])
                line_graph_path_labels.extend([0, 0])
                mol_graph_path_labels.extend([0, 0])
                virtual_path_labels.extend([n + 1, n + 1])
                self_loop_labels.extend([0, 0])
            atom_pairs_features_in_triplets.append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
            virtual_atom_and_virtual_node_labels.append(n + 1)

        if self.add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i, i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(0)
                virtual_path_labels.append(0)
                self_loop_labels.append(1)
        edges = np.array(edges, dtype=np.int64)
        data = (edges[:, 0], edges[:, 1])
        g = dgl.graph(data)
        g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
        g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
        g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
        g.edata['path'] = torch.LongTensor(paths)
        g.edata['lgp'] = torch.BoolTensor(line_graph_path_labels)
        g.edata['mgp'] = torch.BoolTensor(mol_graph_path_labels)
        g.edata['vp'] = torch.BoolTensor(virtual_path_labels)
        g.edata['sl'] = torch.BoolTensor(self_loop_labels)
        return g

    def _generate_augmented_graphs(self, base_smiles):
        two_mer, query_dict = self._build_bond_mapping(base_smiles)
        graphs = []
        smiless = periodicity_augment_traverse(base_smiles)

        processed_smiless = []
        processed_smiless_nmrus = []
        for smiles in smiless:
            for i in range(1, 2):
                processed_smiless.append(generate_multimer_smiles(num_repeat_units=i, smiles=smiles, replace_dummy_atoms=False))
                processed_smiless_nmrus.append(i)

        smiless, smiless_nmrus = self._random_select(processed_smiless, processed_smiless_nmrus, 10)

        error_idx = []
        for i, pa in enumerate(smiless):
            try:
                graphs.append(self._build_graph_with_bid(pa, two_mer, query_dict, is_base=(i == 0)))

            except IndexError:
                error_idx.append(i)

        # delete error smiles
        smiless = [smiles for i, smiles in enumerate(smiless) if i not in error_idx]
        smiless_nmrus = [smiles_nmrus for i, smiles_nmrus in enumerate(smiless_nmrus) if i not in error_idx]
        assert len(graphs) == len(smiless)
        return graphs, smiless, smiless_nmrus

    def _build_graph_with_bid(self, smiles: str, two_mer, query_dict, is_base: bool):
        d_atom_feats = 138
        d_bond_feats = 14
        # Canonicalize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
        mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
        modified_mol = self._modify(mol)

        patt = Chem.MolFromSmarts(smiles)
        patt = Chem.rdmolops.RenumberAtoms(patt, new_order)
        modified_patt = self._modify(patt)
        hit_at = self.matcher.match(two_mer, patt if is_base else modified_patt)

        n_atoms = mol.GetNumAtoms()
        atom_features = []
        for atom_id in range(n_atoms):
            atom = mol.GetAtomWithIdx(atom_id)
            atom_features.append(self.atom_featurizer(atom))
        atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan

        # Construct and Featurize Triplet Nodes
        ## bonded atoms
        virtual_atom_and_virtual_node_labels = []

        atom_pairs_features_in_triplets = []
        bond_features_in_triplets = []
        bond_idx_in_triplets = []

        bonded_atoms = set()
        triplet_id = 0

        bonds = list(mol.GetBonds())
        sorted_bonds = sorted(bonds, key=lambda bond: bond.GetIdx())
        if is_base:
            modified_bonds = list(mol.GetBonds())
            sorted_modified_bonds = sorted(modified_bonds, key=lambda bond: bond.GetIdx())
        else:
            modified_bonds = list(modified_mol.GetBonds())
            sorted_modified_bonds = sorted(modified_bonds, key=lambda bond: bond.GetIdx())

        for bond, modified_bond in zip(sorted_bonds, sorted_modified_bonds):
            begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
            atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
            bond_feature = self.bond_featurizer(bond)
            bond_features_in_triplets.append(bond_feature)

            aid1 = hit_at[modified_bond.GetBeginAtomIdx()]
            aid2 = hit_at[modified_bond.GetEndAtomIdx()]
            bond_idx = two_mer.GetBondBetweenAtoms(aid1, aid2).GetIdx()
            bond_idx_in_triplets.append(query_dict[bond_idx])

            bonded_atoms.add(begin_atom_id)
            bonded_atoms.add(end_atom_id)
            virtual_atom_and_virtual_node_labels.append(0)
            atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[end_atom_id, begin_atom_id] = triplet_id
            triplet_id += 1

        ## unbonded atoms
        for atom_id in range(n_atoms):
            if atom_id not in bonded_atoms:
                atom_pairs_features_in_triplets.append(
                    [atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
                bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
                virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)
        # Construct and Featurize Paths between Triplets
        ## line graph paths
        edges = []
        paths = []
        line_graph_path_labels = []
        mol_graph_path_labels = []
        virtual_path_labels = []
        self_loop_labels = []
        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)
                new_paths = [[new_edge[0]] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [new_edge[1]] for new_edge in new_edges]
                paths.extend(new_paths)
                n_new_edges = len(new_edges)
                line_graph_path_labels.extend([1] * n_new_edges)
                mol_graph_path_labels.extend([0] * n_new_edges)
                virtual_path_labels.extend([0] * n_new_edges)
                self_loop_labels.extend([0] * n_new_edges)

        # # molecule graph paths
        adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
        nx_g = nx.from_numpy_array(adj_matrix)
        paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, self.max_length + 1))
        for i in paths_dict.keys():
            for j in paths_dict[i]:
                path = paths_dict[i][j]
                path_length = len(path)
                if 3 < path_length <= self.max_length + 1:
                    triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]] for pi in range(len(path) - 1)]
                    path_start_triplet_id = triplet_ids[0]
                    path_end_triplet_id = triplet_ids[-1]
                    triplet_path = triplet_ids[1:-1]
                    # assert [path_start_triplet_id,path_end_triplet_id] not in edges
                    triplet_path = [path_start_triplet_id] + triplet_path + [VIRTUAL_PATH_INDICATOR] * (
                                self.max_length - len(triplet_path) - 2) + [path_end_triplet_id]
                    paths.append(triplet_path)
                    edges.append([path_start_triplet_id, path_end_triplet_id])
                    line_graph_path_labels.append(0)
                    mol_graph_path_labels.append(1)
                    virtual_path_labels.append(0)
                    self_loop_labels.append(0)

        for bid, n in enumerate(range(self.n_virtual_nodes), 1):
            for i in range(len(atom_pairs_features_in_triplets) - n):
                edges.append([len(atom_pairs_features_in_triplets), i])
                edges.append([i, len(atom_pairs_features_in_triplets)])
                paths.append([len(atom_pairs_features_in_triplets)] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [i])
                paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (self.max_length - 2) + [len(atom_pairs_features_in_triplets)])
                line_graph_path_labels.extend([0, 0])
                mol_graph_path_labels.extend([0, 0])
                virtual_path_labels.extend([n + 1, n + 1])
                self_loop_labels.extend([0, 0])
            atom_pairs_features_in_triplets.append([[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
            bond_idx_in_triplets.append(-bid)
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
            virtual_atom_and_virtual_node_labels.append(n + 1)

        if self.add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i, i])
                paths.append([i]+[VIRTUAL_PATH_INDICATOR]*(self.max_length-2)+[i])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(0)
                virtual_path_labels.append(0)
                self_loop_labels.append(1)

        edges = np.array(edges, dtype=np.int64)
        data = (edges[:, 0], edges[:, 1])
        g = dgl.graph(data)
        g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
        g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
        g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
        g.ndata['bid'] = torch.LongTensor(bond_idx_in_triplets)
        g.edata['path'] = torch.LongTensor(paths)
        g.edata['lgp'] = torch.BoolTensor(line_graph_path_labels)
        g.edata['mgp'] = torch.BoolTensor(mol_graph_path_labels)
        g.edata['vp'] = torch.BoolTensor(virtual_path_labels)
        g.edata['sl'] = torch.BoolTensor(self_loop_labels)
        return g

    def _get_model(self):
        config = load_config(self.args)
        vocab = Vocab()
        model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_fp_feats=1191,
            d_md_feats=1613,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=0,
            feat_drop=0,
            n_node_types=vocab.vocab_size
        )

        model = model.to(torch.device(self.device))
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(self.args.model_path).items()})
        return model

    # ------------------------ Static functions ------------------------
    @staticmethod
    def _map_n_to_units(n: int) -> int:
        mapping = {1: 3, 2: 6, 3: 9}
        return mapping.get(int(n), 3)

    @staticmethod
    def _random_select(items, items_pair, n):
        if len(items) != len(items_pair):
            raise ValueError("len(items) must be same as len(items_pair)")

        if len(items) <= n:
            return items, items_pair
        else:
            chosen_indices = random.sample(range(1, len(items)), n - 1)
            chosen_items = [items[0]] + [items[i] for i in chosen_indices]
            chosen_items_pair = [items_pair[0]] + [items_pair[i] for i in chosen_indices]
            return chosen_items, chosen_items_pair

    @staticmethod
    def _build_bond_mapping(smiles: str):
        monomer = Chem.MolFromSmiles(smiles)
        repeat_points = []
        cnct_points = []
        bond_type = []
        num_dummy_atoms = 0
        for atom in monomer.GetAtoms():
            if atom.GetSymbol() == '*':
                repeat_points.append(atom.GetIdx())
                neis = atom.GetNeighbors()
                assert len(neis) == 1, f"*atom has more than one neighbor: {smiles}"
                cnct_points.append(atom.GetNeighbors()[0].GetIdx())
                bond_type.append(
                    monomer.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()).GetBondType())
                num_dummy_atoms += 1

        assert num_dummy_atoms == 2, "molecule has more than 2 *atoms"
        assert bond_type[0] == bond_type[1], "bond type of 2 *atoms are not same"

        num_atoms = monomer.GetNumAtoms()
        num_bonds = monomer.GetNumBonds()
        two_mer = Chem.CombineMols(monomer, monomer)
        trimer = Chem.CombineMols(two_mer, monomer)
        fourmer = Chem.CombineMols(trimer, monomer)
        fivemer = Chem.CombineMols(fourmer, monomer)

        # create index list
        REPEAT_LIST, CNCT_LIST = np.zeros([5, 2]), np.zeros([5, 2])
        for i in range(5):
            REPEAT_LIST[i], CNCT_LIST[i] = np.array(repeat_points) + i * num_atoms, np.array(
                cnct_points) + i * num_atoms

        # add single bond between monomers
        ed_oligomer = Chem.EditableMol(fivemer)
        ed_oligomer.AddBond(int(CNCT_LIST[0, 1]), int(CNCT_LIST[1, 0]), order=bond_type[0])
        ed_oligomer.AddBond(int(CNCT_LIST[1, 1]), int(CNCT_LIST[2, 0]), order=bond_type[0])
        ed_oligomer.AddBond(int(CNCT_LIST[2, 1]), int(CNCT_LIST[3, 0]), order=bond_type[0])
        ed_oligomer.AddBond(int(CNCT_LIST[3, 1]), int(CNCT_LIST[4, 0]), order=bond_type[0])

        final_mol = ed_oligomer.GetMol()
        try:
            final_mol = Chem.RemoveHs(final_mol)
        except:
            pass

        equivalent_bond_idx = []
        for bond in final_mol.GetBonds():
            if bond.GetBeginAtom().GetSymbol() == '*' or bond.GetEndAtom().GetSymbol() == '*':
                equivalent_bond_idx.append(bond.GetIdx())

        dic = {i: i for i in range(num_bonds)}
        for i in range(num_bonds, 2 * num_bonds):
            dic[i] = i - num_bonds
        for i in range(2 * num_bonds, 3 * num_bonds):
            dic[i] = i - 2 * num_bonds
        for i in range(3 * num_bonds, 4 * num_bonds):
            dic[i] = i - 3 * num_bonds
        for i in range(4 * num_bonds, 5 * num_bonds):
            dic[i] = i - 4 * num_bonds
        dic[5 * num_bonds] = 5 * num_bonds
        dic[5 * num_bonds + 1] = 5 * num_bonds
        dic[5 * num_bonds + 2] = 5 * num_bonds
        dic[5 * num_bonds + 3] = 5 * num_bonds
        for i in equivalent_bond_idx:
            dic[i] = 5 * num_bonds

        return final_mol, dic

    @staticmethod
    def _modify(mol):
        emol = Chem.EditableMol(mol)
        dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
        if len(dummy_atoms) != 2:
            raise ValueError("Molecule must have exactly two '*' atoms")

        for i, dummy_atom in enumerate(dummy_atoms):
            other_dummy_atom = dummy_atoms[1 - i]
            neighbors = other_dummy_atom.GetNeighbors()
            if not neighbors:
                continue

            neighbor_atom = neighbors[0]
            emol.ReplaceAtom(dummy_atom.GetIdx(), Chem.Atom(neighbor_atom.GetSymbol()))
        return emol.GetMol()
