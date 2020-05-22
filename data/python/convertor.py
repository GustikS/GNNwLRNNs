import sys

# from features import *
# from rdkit import Chem

import numpy as np

class Molecule:

    def __init__(self, all_atom_types, all_bond_types):
        self.all_bond_types = all_bond_types
        self.all_atom_types = all_atom_types
        self.atom_idx = {}
        self.bond_idx = {}

    def load_from_lists(self, atom_types, bond_types, bonds, target):
        self.atom_types = {k: v for k, v in atom_types}
        self.bond_types = {k: v for k, v in bond_types}
        self.bonds = {k: (v, w) for v, w, k in bonds}
        self.target = target

    def to_graph(self):
        graph = {}

        atom_feats = []
        bond_feats = []

        for i, (atom, type) in enumerate(self.atom_types.items()):
            self.atom_idx[atom] = i
            atom_feats.append(to_onehot(self.all_atom_types[type], len(self.all_atom_types)))

        for i, (bond, type) in enumerate(self.bond_types.items()):
            self.bond_idx[bond] = i
            bond_feats.append(to_onehot(self.all_bond_types[type], len(self.all_bond_types)))

        edges_list = [None] * len(self.bonds)
        for i, (bond, (a1, a2)) in enumerate(self.bonds.items()):
            edges_list[self.bond_idx[bond]] = (self.atom_idx[a1], self.atom_idx[a2])

        graph["edge_index"] = np.array(edges_list, dtype=np.int64).T
        graph['edge_feat'] = np.array(bond_feats, dtype=np.float)
        graph['node_feat'] = np.array(atom_feats, dtype=np.float)

        graph['num_nodes'] = len(atom_feats)

        graph["target"] = self.target

        return graph


def to_onehot(ind, size):
    listofzeros = [0] * size
    listofzeros[ind] = 1
    return listofzeros


def graph2prolog(graph):
    bonds = get_bonds(graph["edge_index"].transpose().tolist())
    bond_feats, b_embeds = get_bond_features(graph["edge_feat"].tolist())  # ,graph['bond_types'])
    atom_feats, a_embeds = get_atom_features(graph["node_feat"].tolist())  # , graph['atom_types'])
    all_feats = []
    all_feats.extend(bonds)
    all_feats.extend(bond_feats)
    all_feats.extend(atom_feats)
    string = ",".join(all_feats)
    string += "."
    all_embeds = set()
    all_embeds.update(a_embeds)
    all_embeds.update(b_embeds)

    if graph['target']:
        string = graph['target'].replace("\n", "").replace("-1", "0").replace("+1", "1") + " predict :- " + string

    return string, all_embeds


def get_bonds(pairList):
    bond = 0
    bonds = []
    for ind, (i, j) in enumerate(pairList):

        if ind % 2 == 0:
            suff = "l"
            bond += 1
        else:
            suff = "r"

        next = f' bond(a{i}, a{j}, b{bond}{suff})'
        bonds.append(next)
    return bonds


def get_bond_features(triplesList, bond_types=None):
    bond = 0
    bonds = set()

    embeds = set()
    i = 0

    for ind, featrs in enumerate(triplesList):
        if ind % 2 == 0:
            suff = "l"
            bond += 1
        else:
            suff = "r"

        next = f' <[{", ".join(str(x) for x in featrs)}]> b_feats(b{bond}{suff})'
        bonds.add(next)

        if bond_types:
            type = "bondic" + bond_types[i]
            bonds.add(type + f'(b{bond}{suff})')
            embeds.add(type)

    return bonds, embeds


def get_atom_features(featureList, atom_types=None):
    atom = 0
    atoms = set()

    embeds = set()

    if atom_types:
        for i, featrs in enumerate(featureList):
            type = "atomic" + atom_types[i]
            embeds.add(type)
            next = type + "(a" + str(i) + ")"
            atoms.add(next)

    for i, featrs in enumerate(featureList):
        next = " <[" + ', '.join(str(x) for x in featrs) + "]> a_feats(a" + str(i) + ")"
        atoms.add(next)
    return atoms, embeds
