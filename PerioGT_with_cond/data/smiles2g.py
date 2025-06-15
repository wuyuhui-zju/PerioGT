import torch
import dgl
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from dgllife.utils.featurizers import ConcatFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, \
    bond_stereo_one_hot, atomic_number_one_hot, atom_degree_one_hot, atom_formal_charge, \
    atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_is_aromatic, atom_total_num_H_one_hot, \
    atom_is_chiral_center, atom_chirality_type_one_hot, atom_mass
from mordred import Calculator, descriptors
from functools import partial
from itertools import permutations
import networkx as nx

from utils.aug import generate_oligomer_smiles, knowledge_augment_traverse


__all__ = ["smiles_to_graph_with_prompt"]


MAX_ITERATION_TIME = 3
INF = 1e6
VIRTUAL_ATOM_INDICATOR = -1
VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1
VIRTUAL_PATH_INDICATOR = -INF

d_atom_feats = 138
d_bond_feats = 14
N_ATOM_TYPES = 102
N_BOND_TYPES = 5
bond_featurizer_all = ConcatFeaturizer([  # 14
    partial(bond_type_one_hot, encode_unknown=True),  # 5
    bond_is_conjugated,  # 1
    bond_is_in_ring,  # 1
    partial(bond_stereo_one_hot, encode_unknown=True)  # 7
])
atom_featurizer_all = ConcatFeaturizer([  # 137
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


def smiles_to_prompt(base_smiles, model, scaler, device, max_length, n_virtual_nodes):
    graphs, smiless, smiless_n = smiles_to_graph_node_emb(base_smiles, max_length, n_virtual_nodes)
    base_graph = graphs[0]

    fp_with_n = []
    des_with_n = []
    calc = Calculator(descriptors, ignore_3D=True)
    for i in [3, 6, 9]:
        smiles = generate_oligomer_smiles(num_repeat_units=i, smiles=base_smiles)
        mol = Chem.MolFromSmiles(smiles)
        # fp
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        ec_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        # md
        des = np.array(list(calc(mol).values()), dtype=np.float32)

        fp_with_n.append(list(map(int, list(maccs_fp + ec_fp))))
        des_with_n.append(des)

    fp_list = []
    des_list = []
    for n in smiless_n:
        fp_list.append(fp_with_n[n - 1])
        des_list.append(des_with_n[n - 1])

    fp_list = np.array(fp_list, dtype=np.float32)
    des = np.array(des_list, dtype=np.float32)
    des = np.where(np.isnan(des), 0, des)
    des = np.where(des > 10 ** 12, 10 ** 12, des)

    # normalization
    des_norm = scaler.transform(des).astype(np.float32)

    graphs = dgl.batch(graphs).to(device)
    fps = torch.from_numpy(fp_list).to(device)
    mds = torch.from_numpy(des_norm).to(device)

    node_embs, bids = model.generate_node_emb(graphs, fps, mds)
    bids = bids.cpu().numpy().tolist()
    BID_to_PROMPT_DICT = {bid: [] for bid in bids}
    for i, bid in enumerate(bids):
        prompt = node_embs[i]
        BID_to_PROMPT_DICT[bid].append(prompt.detach().cpu())

    max_node = 20
    for k, v in BID_to_PROMPT_DICT.items():
        # if len(v) > max_node:
        #     max_node = len(v)
        BID_to_PROMPT_DICT[k] = torch.stack(v)

    feats = torch.zeros([base_graph.number_of_nodes(), 20, BID_to_PROMPT_DICT[-1].size()[1]])
    for nid in range(base_graph.number_of_nodes()):
        bid = base_graph.ndata['bid'][nid].item()
        feat = BID_to_PROMPT_DICT[bid]
        feats[nid][:feat.size()[0], :] = feat[:20]

    return feats


def smiles_to_graph_node_emb(smiles_base, max_length=5, n_virtual_nodes=2, add_self_loop=True):
    two_mer, query_dict = generate_idx_dict(smiles_base)
    graphs = []
    smiless = knowledge_augment_traverse(smiles_base)

    processed_smiless = []
    processed_smiless_nmrus = []
    for smiles in smiless:
        for i in range(1, 4):
            processed_smiless.append(generate_oligomer_smiles(num_repeat_units=i, smiles=smiles, replace_star_atom=False))
            processed_smiless_nmrus.append(i)

    def random_select(items, items_pair, n):
        if len(items) != len(items_pair):
            raise ValueError("len(items) must be same as len(items_pair)")

        if len(items) <= n:
            return items, items_pair
        else:
            chosen_indices = random.sample(range(1, len(items)), n - 1)
            chosen_items = [items[0]] + [items[i] for i in chosen_indices]
            chosen_items_pair = [items_pair[0]] + [items_pair[i] for i in chosen_indices]
            return chosen_items, chosen_items_pair

    smiless, smiless_nmrus = random_select(processed_smiless, processed_smiless_nmrus, 10)

    error_idx = []
    for i, ka in enumerate(smiless):
        try:
            if i == 0:
                graphs.append(node_emb(ka, two_mer, query_dict, max_length, n_virtual_nodes, add_self_loop, is_base=True))
            else:
                graphs.append(node_emb(ka, two_mer, query_dict, max_length, n_virtual_nodes, add_self_loop, is_base=False))

        except IndexError:
            error_idx.append(i)
            print(f"Can't match substructure: {ka}")
            print(f"Base smiles: {smiles_base}")

    # delete error smiles
    smiless = [smiles for i, smiles in enumerate(smiless) if i not in error_idx]
    smiless_nmrus = [smiles_nmrus for i, smiles_nmrus in enumerate(smiless_nmrus) if i not in error_idx]

    assert len(graphs) == len(smiless)

    return graphs, smiless, smiless_nmrus


def node_emb(smiles, two_mer, query_dict, max_length, n_virtual_nodes, add_self_loop, is_base):
    """
        This is a helper function of smiles_to_graph_node_emb()
    """
    d_atom_feats = 138
    d_bond_feats = 14
    # Canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
    modified_mol = modify(mol)
    patt = Chem.MolFromSmarts(smiles)
    patt = Chem.rdmolops.RenumberAtoms(patt, new_order)
    modified_patt = modify(patt)
    # gen_bond_idx_dict
    if is_base:
        hit_at = two_mer.GetSubstructMatch(patt)
    else:
        hit_at = two_mer.GetSubstructMatch(modified_patt)


    # print(Chem.MolToSmiles(modified_patt))
    # Featurize Atoms
    n_atoms = mol.GetNumAtoms()
    atom_features = []

    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_features.append(atom_featurizer_all(atom))
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
        bond_feature = bond_featurizer_all(bond)
        bond_features_in_triplets.append(bond_feature)

        aid1 = hit_at[modified_bond.GetBeginAtomIdx()]
        aid2 = hit_at[modified_bond.GetEndAtomIdx()]
        bond_idx = two_mer.GetBondBetweenAtoms(aid1, aid2).GetIdx()
        bond_idx_in_triplets.append(query_dict[bond_idx])

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
            new_paths = [[new_edge[0]] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [new_edge[1]] for new_edge in
                         new_edges]
            paths.extend(new_paths)
            n_new_edges = len(new_edges)
            line_graph_path_labels.extend([1] * n_new_edges)
            mol_graph_path_labels.extend([0] * n_new_edges)
            virtual_path_labels.extend([0] * n_new_edges)
            self_loop_labels.extend([0] * n_new_edges)
    # # molecule graph paths
    adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
    nx_g = nx.from_numpy_array(adj_matrix)
    paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, max_length + 1))
    for i in paths_dict.keys():
        for j in paths_dict[i]:
            path = paths_dict[i][j]
            path_length = len(path)
            if 3 < path_length <= max_length + 1:
                triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi + 1]] for pi in range(len(path) - 1)]
                path_start_triplet_id = triplet_ids[0]
                path_end_triplet_id = triplet_ids[-1]
                triplet_path = triplet_ids[1:-1]
                # assert [path_start_triplet_id,path_end_triplet_id] not in edges
                triplet_path = [path_start_triplet_id] + triplet_path + [VIRTUAL_PATH_INDICATOR] * (
                        max_length - len(triplet_path) - 2) + [path_end_triplet_id]
                paths.append(triplet_path)
                edges.append([path_start_triplet_id, path_end_triplet_id])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(1)
                virtual_path_labels.append(0)
                self_loop_labels.append(0)
    for bid, n in enumerate(range(n_virtual_nodes), 1):
        for i in range(len(atom_pairs_features_in_triplets) - n):
            edges.append([len(atom_pairs_features_in_triplets), i])
            edges.append([i, len(atom_pairs_features_in_triplets)])
            paths.append([len(atom_pairs_features_in_triplets)] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [len(atom_pairs_features_in_triplets)])
            line_graph_path_labels.extend([0, 0])
            mol_graph_path_labels.extend([0, 0])
            virtual_path_labels.extend([n + 1, n + 1])
            self_loop_labels.extend([0, 0])
        atom_pairs_features_in_triplets.append(
            [[VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats])
        bond_idx_in_triplets.append(-bid)
        bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats)
        virtual_atom_and_virtual_node_labels.append(n + 1)
    if add_self_loop:
        for i in range(len(atom_pairs_features_in_triplets)):
            edges.append([i, i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i])
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


def modify(mol):
    emol = Chem.EditableMol(mol)

    wildcard_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(wildcard_atoms) != 2:
        raise ValueError("Molecule must have exactly two '*' atoms")

    for i, wildcard_atom in enumerate(wildcard_atoms):
        other_wildcard_atom = wildcard_atoms[1 - i]
        neighbors = other_wildcard_atom.GetNeighbors()
        if not neighbors:
            continue

        neighbor_atom = neighbors[0]
        emol.ReplaceAtom(wildcard_atom.GetIdx(), Chem.Atom(neighbor_atom.GetSymbol()))

    return emol.GetMol()


def generate_idx_dict(smiles):
    """
    This is a helper function of smiles_to_graph_node_emb()
    """
    monomer = Chem.MolFromSmiles(smiles)
    repeat_points = []
    cnct_points = []
    bond_type = []
    num_star_atoms = 0
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            repeat_points.append(atom.GetIdx())
            neis = atom.GetNeighbors()
            assert len(neis) == 1, f"*atom has more than one neighbor: {smiles}"
            cnct_points.append(atom.GetNeighbors()[0].GetIdx())
            bond_type.append(monomer.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()).GetBondType())
            num_star_atoms += 1

    assert num_star_atoms == 2, "molecule has more than 2 *atoms"
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
        REPEAT_LIST[i], CNCT_LIST[i] = np.array(repeat_points) + i * num_atoms, np.array(cnct_points) + i * num_atoms

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


def construct_single_component(smiles, max_length=5, n_knowledge_nodes=2, start_idx=0, index=0, add_self_loop=True):

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
        atom_features.append(atom_featurizer_all(atom))
    atomIDPair_to_tripletId = np.ones(shape=(n_atoms,n_atoms))*np.nan
    # Construct and Featurize Triplet Nodes
    ## bonded atoms
    virtual_atom_and_virtual_node_labels = []

    atom_pairs_features_in_triplets = []
    bond_features_in_triplets = []

    bonded_atoms = set()
    triplet_id = start_idx
    for bond in mol.GetBonds():
        begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
        atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
        bond_feature = bond_featurizer_all(bond)
        bond_features_in_triplets.append(bond_feature)
        bonded_atoms.add(begin_atom_id)
        bonded_atoms.add(end_atom_id)
        virtual_atom_and_virtual_node_labels.append(0)
        atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[end_atom_id, begin_atom_id] = triplet_id
        triplet_id += 1
    ## unbonded atoms
    for atom_id in range(n_atoms):
        if atom_id not in bonded_atoms:
            atom_pairs_features_in_triplets.append([atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER]*d_bond_feats)
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
            new_edges = list(permutations(node_ids,2))
            edges.extend(new_edges)
            new_paths = [[new_edge[0]]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[new_edge[1]] for new_edge in new_edges]
            paths.extend(new_paths)
            n_new_edges = len(new_edges)
            line_graph_path_labels.extend([1]*n_new_edges)
            mol_graph_path_labels.extend([0]*n_new_edges)
            virtual_path_labels.extend([0]*n_new_edges)
            self_loop_labels.extend([0]*n_new_edges)
    # # molecule graph paths
    adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
    nx_g = nx.from_numpy_array(adj_matrix)
    paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g,max_length+1))
    for i in paths_dict.keys():
        for j in paths_dict[i]:
            path = paths_dict[i][j]
            path_length = len(path)
            if 3 < path_length <= max_length+1:
                triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi+1]] for pi in range(len(path)-1)]
                path_start_triplet_id = triplet_ids[0]
                path_end_triplet_id = triplet_ids[-1]
                triplet_path = triplet_ids[1:-1]
                # assert [path_start_triplet_id,path_end_triplet_id] not in edges
                triplet_path = [path_start_triplet_id]+triplet_path+[VIRTUAL_PATH_INDICATOR]*(max_length-len(triplet_path)-2)+[path_end_triplet_id]
                paths.append(triplet_path)
                edges.append([path_start_triplet_id, path_end_triplet_id])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(1)
                virtual_path_labels.append(0)
                self_loop_labels.append(0)

    for j, n in enumerate(range(index*n_knowledge_nodes, (index+1)*n_knowledge_nodes)):
        for i in range(len(atom_pairs_features_in_triplets)-j):
            edges.append([len(atom_pairs_features_in_triplets)+start_idx, i+start_idx])
            edges.append([i+start_idx, len(atom_pairs_features_in_triplets)+start_idx])
            paths.append([len(atom_pairs_features_in_triplets)+start_idx]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[i+start_idx])
            paths.append([i+start_idx]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[len(atom_pairs_features_in_triplets)+start_idx])
            line_graph_path_labels.extend([0,0])
            mol_graph_path_labels.extend([0,0])
            virtual_path_labels.extend([n+1,n+1])
            self_loop_labels.extend([0,0])
        atom_pairs_features_in_triplets.append([[VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats])
        bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER]*d_bond_feats)
        virtual_atom_and_virtual_node_labels.append(n+1)
    if add_self_loop:
        for i in range(len(atom_pairs_features_in_triplets)):
            edges.append([i+start_idx, i+start_idx])
            paths.append([i+start_idx]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[i+start_idx])
            line_graph_path_labels.append(0)
            mol_graph_path_labels.append(0)
            virtual_path_labels.append(0)
            self_loop_labels.append(1)

    return edges, atom_pairs_features_in_triplets, bond_features_in_triplets, virtual_atom_and_virtual_node_labels, paths, line_graph_path_labels, mol_graph_path_labels, virtual_path_labels, self_loop_labels


def smiles_to_graph_tune(smiles_list, max_length=5, n_knowledge_nodes=2, n_global_nodes=2, add_self_loop=True):
    smiles_list = list(filter(None, smiles_list))
    edges, atom_pairs_features_in_triplets, bond_features_in_triplets, virtual_atom_and_virtual_node_labels, paths, line_graph_path_labels, mol_graph_path_labels, virtual_path_labels, self_loop_labels = [], [], [], [], [], [], [], [], []
    for i, smiles in enumerate(smiles_list):
        edges_new, atom_pairs_features_in_triplets_new, bond_features_in_triplets_new, virtual_atom_and_virtual_node_labels_new, paths_new, line_graph_path_labels_new, mol_graph_path_labels_new, virtual_path_labels_new, self_loop_labels_new = construct_single_component(smiles, max_length=5, n_knowledge_nodes=n_knowledge_nodes, add_self_loop=True, start_idx=max([v for edge in edges for v in edge])+1 if len(edges) > 0 else 0, index=i)
        edges.extend(edges_new)
        atom_pairs_features_in_triplets.extend(atom_pairs_features_in_triplets_new)
        bond_features_in_triplets.extend(bond_features_in_triplets_new)
        virtual_atom_and_virtual_node_labels.extend(virtual_atom_and_virtual_node_labels_new)
        paths.extend(paths_new)
        line_graph_path_labels.extend(line_graph_path_labels_new)
        mol_graph_path_labels.extend(mol_graph_path_labels_new)
        virtual_path_labels.extend(virtual_path_labels_new)
        self_loop_labels.extend(self_loop_labels_new)

    for n in range(n_global_nodes):
        for i in range(len(atom_pairs_features_in_triplets)):
            if virtual_atom_and_virtual_node_labels[i] > 0:
                continue
            edges.append([len(atom_pairs_features_in_triplets), i])
            edges.append([i, len(atom_pairs_features_in_triplets)])
            paths.append([len(atom_pairs_features_in_triplets)]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[i])
            paths.append([i]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[len(atom_pairs_features_in_triplets)])
            line_graph_path_labels.extend([0,0])
            mol_graph_path_labels.extend([0,0])
            virtual_path_labels.extend([n_knowledge_nodes*len(smiles_list)+n+1,n_knowledge_nodes*len(smiles_list)+n+1])
            self_loop_labels.extend([0,0])
        atom_pairs_features_in_triplets.append([[VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats])
        bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER]*d_bond_feats)
        virtual_atom_and_virtual_node_labels.append(n_knowledge_nodes*len(smiles_list)+n+1)

    edges = np.array(edges, dtype=np.int64)
    data = (edges[:,0], edges[:,1])
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


def smiles_to_graph_with_prompt(smiles_lst, model, scaler, device, max_length=5, n_knowledge_nodes=2, n_global_nodes=1):
    g = smiles_to_graph_tune(
        smiles_lst,
        max_length=max_length,
        n_knowledge_nodes=n_knowledge_nodes,
        n_global_nodes=n_global_nodes
    )
    prompt = smiles_to_prompt(smiles_lst[0], model, scaler, device, max_length=max_length, n_virtual_nodes=2)
    feats = torch.zeros([g.number_of_nodes(), 20, prompt.size()[-1]])
    feats[:-1] = prompt

    g.ndata["prompt"] = feats
    return g
