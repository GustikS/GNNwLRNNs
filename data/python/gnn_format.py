import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader, Dataset
from tqdm import tqdm

def _split(data_list, n):
    folds = []
    size = int(len(data_list) / n)
    i = 0
    while i < len(data_list) - size + 1:
        folds.append(data_list[i:i + size])
        i += size
    while i < len(data_list):
        folds[i % n].append(data_list[i])
        i += 1
    return folds


def save_obj(obj, name):
    Path(name).parent.mkdir(exist_ok=True, parents=True)
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def graphs_to_pyg(graph_list):
    pyg_graph_list = []
    print('Converting graphs into PyG objects...')
    for graph in tqdm(graph_list):
        g = Data()
        g.__num_nodes__ = graph["num_nodes"]
        g.edge_index = torch.tensor(graph["edge_index"])

        if graph["edge_feat"] is not None:
            g.edge_attr = torch.DoubleTensor(graph["edge_feat"])

        if graph["node_feat"] is not None:
            g.x = torch.DoubleTensor(graph["node_feat"])

        if graph["target"] is not None:
            tar = int(graph["target"])
            tar = 0 if tar == -1 else tar
            g.y = torch.LongTensor([tar])

        pyg_graph_list.append(g)

    return pyg_graph_list