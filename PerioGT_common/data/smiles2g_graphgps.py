import torch
import dgl
import numpy as np
from rdkit import Chem
from dgllife.utils.featurizers import ConcatFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, \
    bond_stereo_one_hot, atomic_number_one_hot, atom_degree_one_hot, atom_formal_charge, \
    atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_is_aromatic, atom_total_num_H_one_hot, \
    atom_is_chiral_center, atom_chirality_type_one_hot, atom_mass
from functools import partial
from itertools import permutations

from data.constants import INF, VIRTUAL_ATOM_INDICATOR, VIRTUAL_ATOM_FEATURE_PLACEHOLDER,\
    VIRTUAL_BOND_FEATURE_PLACEHOLDER, VIRTUAL_PATH_INDICATOR, N_ATOM_TYPES, N_BOND_TYPES
from utils.function import random_walk_pe


class PolyGraphBuilderPretrain:
    def __init__(self, vocab, pos_enc_size=28, n_virtual_nodes=2, add_self_loop=True):
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
        self.pos_enc_size = pos_enc_size
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
        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)

        for n in range(self.n_virtual_nodes):
            for i in range(len(atom_pairs_features_in_triplets) - n):
                edges.append([len(atom_pairs_features_in_triplets), i])
                edges.append([i, len(atom_pairs_features_in_triplets)])

            atom_pairs_features_in_triplets.append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
            triplet_labels.append(self.vocab.index(999, 999, 999))
            virtual_atom_and_virtual_node_labels.append(n + 1)

        if self.add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i, i])

        edges = np.array(edges, dtype=np.int64)
        data = (edges[:, 0], edges[:, 1])
        g = dgl.graph(data)
        g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
        g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
        g.ndata['label'] = torch.LongTensor(triplet_labels)
        g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
        g.ndata['PE'] = random_walk_pe(g, k=self.pos_enc_size)
        return g


class PolyGraphBuilderFinetune:
    def __init__(self, args, scaler, pos_enc_size=28, n_virtual_nodes=2, add_self_loop=True):
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

        self.pos_enc_size = pos_enc_size
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
        self.use_prompt = self.args.use_prompt

    def build(self, smiles: str, feat_cache=None):
        if self.use_prompt:
            return self._build_with_prompt(smiles, feat_cache=feat_cache)
        else:
            return self._build_without_prompt(smiles)

    def _build_with_prompt(self, smiles: str, feat_cache):
        """
        Placeholder for periodicity prompt guided finetuning for GraphGPS.
        Although GraphGPS was initially considered, it was not used in subsequent experiments
        due to LiGhT demonstrating better robustness in this work.
        This function is kept for structural consistency and potential future extension.
        """
        pass

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
        for i in range(n_atoms):
            node_ids = atomIDPair_to_tripletId[i]
            node_ids = node_ids[~np.isnan(node_ids)]
            if len(node_ids) >= 2:
                new_edges = list(permutations(node_ids, 2))
                edges.extend(new_edges)

        for n in range(self.n_virtual_nodes):
            for i in range(len(atom_pairs_features_in_triplets) - n):
                edges.append([len(atom_pairs_features_in_triplets), i])
                edges.append([i, len(atom_pairs_features_in_triplets)])

            atom_pairs_features_in_triplets.append(
                [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
            virtual_atom_and_virtual_node_labels.append(n + 1)

        if self.add_self_loop:
            for i in range(len(atom_pairs_features_in_triplets)):
                edges.append([i, i])

        edges = np.array(edges, dtype=np.int64)
        data = (edges[:, 0], edges[:, 1])
        g = dgl.graph(data)
        g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
        g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
        g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
        g.ndata['PE'] = random_walk_pe(g, k=self.pos_enc_size)

        return g
