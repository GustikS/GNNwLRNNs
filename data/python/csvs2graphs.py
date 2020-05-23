import os
import sys

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

from gnn_format import save_obj, graphs_to_pyg
import numpy as np

from convertor import graph2prolog, Molecule

all_embeds = set()


def stratified_split(molecules: [Molecule], folds=10):
    skf = StratifiedKFold(n_splits=folds)

    labels = [mol.target for mol in molecules]

    folds = []
    for train_idx, test_idx in skf.split(np.zeros(len(molecules)), labels):
        train_fold_tmp = [molecules[i] for i in train_idx]

        y_train = [mol.target for mol in train_fold_tmp]

        train_fold, val_fold, _, _ = train_test_split(train_fold_tmp, y_train, stratify=y_train,
                                                      test_size=0.1, random_state=1)

        test_fold = [molecules[i] for i in test_idx]
        folds.append((train_fold, val_fold, test_fold))
    return folds


class CsvMolecule:

    def __init__(self, atom_types, bond_types, bonds, target):
        self.target = target
        self.bonds = bonds
        self.bond_types = bond_types
        self.atom_types = atom_types


def mol_iterator(lines):
    i = 0
    while i < len(lines):
        next = []
        while not "---" in lines[i]:
            next.append(lines[i])
            i += 1
        i += 1
        yield next


def save_graphs_fold(dataset, i, train, val, test, out_path):
    train_graphs = graphs_to_pyg([mol.to_graph() for mol in train])
    save_obj(train_graphs, os.path.join(out_path, dataset, "fold" + str(i), "train_graphs"))
    train_labs = [str(graph.y) for graph in train_graphs]
    save_list(dataset, i, train_labs, "train_labels", out_path)
    val_graphs = graphs_to_pyg([mol.to_graph() for mol in val])
    save_obj(val_graphs, os.path.join(out_path, dataset, "fold" + str(i), "val_graphs"))
    val_labs = [str(graph.y) for graph in val_graphs]
    save_list(dataset, i, val_labs, "val_labels", out_path)
    test_graphs = graphs_to_pyg([mol.to_graph() for mol in test])
    save_obj(test_graphs, os.path.join(out_path, dataset, "fold" + str(i), "test_graphs"))
    test_labs = [str(graph.y) for graph in test_graphs]
    save_list(dataset, i, test_labs, "test_labels", out_path)


def save_lrnn_fold(dataset, i, train, val, test, out_path):
    train_examples = ([graph2prolog(mol.to_graph())[0] for mol in train])
    save_list(dataset, i, train_examples, "trainExamples", out_path)
    val_examples = ([graph2prolog(mol.to_graph())[0] for mol in val])
    save_list(dataset, i, val_examples, "valExamples", out_path)
    test_examples = ([graph2prolog(mol.to_graph())[0] for mol in test])
    save_list(dataset, i, test_examples, "testExamples", out_path)


def save_list(dataset, i, train_examples, name, out_path):
    file = os.path.join(out_path, dataset, "fold" + str(i), name + ".txt")
    Path(file).parent.mkdir(exist_ok=True, parents=True)
    with open(file, "w") as f:
        f.write("\n".join(train_examples))
        f.close()


def convert(in_path, out_path, all_atom_types, all_bond_types):
    for ind, dataset in enumerate(sorted(os.listdir(in_path))):
        if not os.path.isdir(os.path.join(in_path, dataset)):
            continue

        print(dataset)
        molecules = []
        csv_molecules = []

        with open(os.path.join(in_path, dataset, "target.txt"), "r") as f:
            targets = f.read().split("\n")

        with open(os.path.join(in_path, dataset, "atom_type.csv"), "r") as f:
            atom_types = f.readlines()

        with open(os.path.join(in_path, dataset, "bond_type.csv"), "r") as f:
            bond_types = f.readlines()

        with open(os.path.join(in_path, dataset, "bond.csv"), "r") as f:
            bond_lines = f.readlines()

        for a_types, b_types, bonds, target in zip(mol_iterator(atom_types), mol_iterator(bond_types),
                                                   mol_iterator(bond_lines), targets):
            mol = Molecule(all_atom_types, all_bond_types)

            mol.load_from_lists([a.split(";")[1:3] for a in a_types], [a.split(";")[1:3] for a in b_types],
                                [a.split(";")[1:4] for a in bonds], target)
            molecules.append(mol)
            csv_molecules.append(CsvMolecule(a_types, b_types, bonds, target))  # todo cover within Molecule class

        folds = stratified_split(molecules, 10)

        for i, (train, val, test) in enumerate(folds):
            save_graphs_fold(dataset, i, train, val, test, out_path)
            save_lrnn_fold(dataset, i, train, val, test, out_path)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print("Provide 2 arguments, path to 1) folder with transformed molecular datasets, 2) output folder")

    in_path = sys.argv[1]
    if len(sys.argv) == 2:
        out_path = os.path.join(in_path, "splits")
    else:
        out_path = sys.argv[2]

    all_atom_types = {}
    with open(os.path.join(in_path, "atom_types.txt")) as f:
        lines = f.read().split("\n")
        for i, line in enumerate(lines):
            all_atom_types[line] = i

    all_bond_types = {}
    with open(os.path.join(in_path, "bond_types.txt")) as f:
        lines = f.read().split("\n")
        for i, line in enumerate(lines):
            all_bond_types[line] = i

    convert(in_path, out_path, all_atom_types, all_bond_types)
