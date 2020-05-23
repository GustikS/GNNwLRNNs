import json
import pickle
import statistics
import time
import argparse
from os import listdir

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, avg_pool, global_mean_pool, JumpingKnowledge, global_add_pool, GINConv, SAGEConv

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from random import shuffle

from tensorboardX import SummaryWriter

torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class NetGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dim=10):
        super(NetGCN, self).__init__()

        self.conv1 = GCNConv(num_features, dim, normalize=False, cached=False, bias=False)
        self.conv2 = GCNConv(dim, dim, normalize=False, cached=False, bias=False)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

        self.fc1 = Linear(dim, 1, bias=False)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        return torch.sigmoid(x)


class NetGraphSage(torch.nn.Module):
    def __init__(self, num_features, num_classes, concat=False, dim=10):
        super(NetGraphSage, self).__init__()
        self.conv1 = SAGEConv(num_features, dim, normalize=False, concat=True, bias=False)
        self.conv2 = SAGEConv(dim, dim, normalize=False, concat=True, bias=False)

        self.fc1 = Linear(dim, 1, bias=False)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        return torch.sigmoid(x)


class NetGIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, dim=10):
        super(NetGIN, self).__init__()

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)

        self.l1 = Linear(dim, 1, bias=False)
        self.l2 = Linear(dim, 1, bias=False)
        self.l3 = Linear(dim, 1, bias=False)
        self.l4 = Linear(dim, 1, bias=False)
        self.l5 = Linear(dim, 1, bias=False)

    def forward(self, x, edge_index, batch):
        x1 = F.relu(self.conv1(x, edge_index))
        # x = self.bn1(x)
        x2 = F.relu(self.conv2(x1, edge_index))
        # x = self.bn2(x)
        x3 = F.relu(self.conv3(x2, edge_index))
        # x = self.bn3(x)
        x4 = F.relu(self.conv4(x3, edge_index))
        # x = self.bn4(x)
        x5 = F.relu(self.conv5(x4, edge_index))
        # x = self.bn5(x)
        m1 = global_mean_pool(x1, batch)
        m2 = global_mean_pool(x2, batch)
        m3 = global_mean_pool(x3, batch)
        m4 = global_mean_pool(x4, batch)
        m5 = global_mean_pool(x5, batch)

        suma = torch.stack([self.l1(m1),
                            self.l2(m2),
                            self.l3(m3),
                            self.l4(m4),
                            self.l5(m5)], dim=0)

        x = torch.sum(suma, dim=0)

        return torch.sigmoid(x)


# %%
class Crossval:
    def __init__(self, train_results, val_results, test_results, times=[]):
        self.train_loss_mean, self.train_loss_std = self.aggregate_loss(train_results)
        self.train_acc_mean, self.train_acc_std = self.aggregate_acc(train_results)

        self.val_loss_mean, self.val_loss_std = self.aggregate_loss(val_results)
        self.val_acc_mean, self.val_acc_std = self.aggregate_acc(val_results)

        self.test_loss_mean, self.test_loss_std = self.aggregate_loss(test_results)
        self.test_acc_mean, self.test_acc_std = self.aggregate_acc(test_results)

        if times:
            self.time_per_step = sum(times) / len(times)

    def aggregate_loss(self, results):
        losses = [res.loss for res in results]
        return np.mean(losses), np.std(losses)

    def aggregate_acc(self, results):
        accuracies = [res.accuracy for res in results]
        return statistics.mean(accuracies), statistics.stdev(accuracies)


class Results:

    def __init__(self, pred=[], lab=[], loss_fcn=F.binary_cross_entropy):
        self.loss = 0
        self.accuracy = 0

        if pred and lab:
            # pred = [p.detach().numpy() for p in pred]
            # lab = [l.detach().numpy() for l in lab]
            for (p, l) in zip(pred, lab):
                if p.ndim == 1:     # batching changes this
                    self.loss += loss_fcn(p[0], l.double())
                else:
                    self.loss += loss_fcn(p[0][0], l[0].double())
            self.loss /= len(lab)
            self.loss = float(self.loss)
            self.accuracy = self.acc(pred, lab)

    def acc(self, output, labels):
        correct = 0
        for out, lab in zip(output, labels):
            if len(out) == 1:
                out_ = out
            else:
                out_ = out[0][0]

            pred = 1 if out_ > 0.5 else 0

            if lab.ndim == 0:
                lab_ = lab
            else:
                lab_ = lab[0]

            if pred == int(lab_):
                correct += 1
        return correct / len(labels)

    def increment(self, other):
        self.loss = other.loss
        self.accuracy = other.accuracy


class ResultList:
    def __init__(self, results, times=None):
        self.folds = [to_json(res) for res in results]
        if times:
            self.times = times


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load_dataset_folds_external(path, suffix="_graphs.pkl", batch=1):
    folds = []

    for fold in sorted(listdir(path)):
        if fold.startswith("fold"):
            train_fold = load_obj(path + "/" + fold + "/train" + suffix)
            val_fold = load_obj(path + "/" + fold + "/val" + suffix)
            test_fold = load_obj(path + "/" + fold + "/test" + suffix)

            folds.append([DataLoader(train_fold, batch_size=batch), DataLoader(val_fold, batch_size=batch),
                          DataLoader(test_fold, batch_size=batch)])

    return folds


def load_dataset_folds(path, batch=1, folds=10):
    dataset = load_obj(path)
    num_features = dataset[0].num_node_features

    shuffle(dataset)

    skf = StratifiedKFold(n_splits=folds)

    labels = [lab.y.numpy() for lab in dataset]

    folds = []
    for train_idx, test_idx in skf.split(np.zeros(len(dataset)), labels):
        train_fold_tmp = [dataset[i] for i in train_idx]

        y_train = [lab.y.numpy() for lab in train_fold_tmp]

        train_fold, val_fold, _, _ = train_test_split(train_fold_tmp, y_train, stratify=y_train,
                                                      test_size=0.1, random_state=1)

        test_fold = [dataset[i] for i in test_idx]
        folds.append([DataLoader(train_fold, batch_size=batch), DataLoader(val_fold, batch_size=batch),
                      DataLoader(test_fold, batch_size=batch)])
    return folds


def train(model, loader, optimizer, epoch):
    model.train()

    # if epoch % 51 == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    outputs = []
    labels = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        # loss = F.nll_loss(output, data.y)
        loss = F.binary_cross_entropy(output[0][0], data.y[0].double())
        if len(data.y) == 1:
            outputs.append(output)
            labels.append(data.y)
        else:
            outputs.extend(output)
            labels.extend(data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return Results(outputs, labels)


def test(model, loader):
    model.eval()

    correct = 0
    outputs = []
    labels = []
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        # correct += pred.eq(data.y).sum().item()
        if len(data.y) == 1:
            outputs.append(output)
            labels.append(data.y)
        else:
            outputs.extend(output)
            labels.extend(data.y)
    return Results(outputs, labels)


def learn(model, train_loader, val_loader, test_loader, writer, steps=1000, lr=0.000015):
    #
    # input = [dataset[0].x, dataset[0].edge_index]
    # # writer.add_graph(model, input)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_results = Results()
    best_val_results.loss = 1e10
    cumtime = 0

    for epoch in range(0, steps):
        start = time.time()
        train_results = train(model, train_loader, optimizer, epoch)

        # train_loss2, train_acc = test(model, train_loader)
        val_results = test(model, val_loader)
        test_results = test(model, test_loader)

        if val_results.loss < best_val_results.loss:
            print(f'improving validation loss to {val_results.loss} at epoch {epoch}')
            best_val_results = val_results
            best_test_results = test_results
            best_train_results = train_results
            print(f'storing respective test results with accuracy {best_test_results.accuracy}')

        end = time.time()
        elapsed = end - start
        cumtime += elapsed
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_results.loss,
                                                                            train_results.accuracy,
                                                                            val_results.accuracy,
                                                                            test_results.accuracy) + " elapsed: " + str(
            elapsed))

        writer.add_scalar('Loss/train', train_results.loss, epoch)
        writer.add_scalar('Loss/val', val_results.loss, epoch)
        writer.add_scalar('Loss/test', test_results.loss, epoch)
        writer.add_scalar('Accuracy/train', train_results.accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_results.accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_results.accuracy, epoch)
        writer.flush()

    return best_train_results, best_val_results, best_test_results, cumtime / steps


def export_fold(content, outpath):
    with open(outpath + ".json", "w") as f:
        f.writelines(content)
        f.close()


def crossvalidate(model_string, folds, outpath, steps=1000, lr=0.000015, dim=10):
    writer = SummaryWriter(outpath)
    if outpath == None:
        outpath = "./" + writer.logdir

    train_results = []
    val_results = []
    test_results = []
    times = []

    counter = 0

    for train_fold, val_fold, test_fold in folds:
        model = get_model(model_string, dim)
        best_train_results, best_val_results, best_test_results, elapsed = learn(model, train_fold, val_fold, test_fold,
                                                                                 writer,
                                                                                 steps, lr)
        train_results.append(best_train_results)
        val_results.append(best_val_results)
        test_results.append(best_test_results)
        times.append(elapsed)

        train = to_json(ResultList(train_results, times=times))
        export_fold(train, outpath + "/train")
        test = to_json(ResultList(test_results))
        export_fold(test, outpath + "/test")
        counter += 1

    cross = Crossval(train_results, val_results, test_results, times)
    return cross, writer


def get_model(string, dim):
    if string == "gcn":
        model = NetGCN(num_node_features, num_classes, dim=dim).to(device)
    elif string == "gin":
        model = NetGIN(num_node_features, num_classes, dim=dim).to(device)
    elif string == "gsage":
        model = NetGraphSage(num_node_features, num_classes, dim=dim).to(device)

    return model


def to_json(obj):
    return json.dumps(obj.__dict__, indent=4)


if __name__ == '__main__':
    num_classes = 2

    parser = argparse.ArgumentParser()

    parser.add_argument("-sd", help="path to dataset for learning", type=str)
    parser.add_argument("-model", help="type of model (gcn,gin)", type=str)
    parser.add_argument("-out", help="path to output folder", type=str)

    parser.add_argument("-xval", nargs='?', help="number fo folds for crossval", type=int)
    parser.add_argument("-lr", nargs='?', help="learning rate for Adam", type=float)
    parser.add_argument("-ts", nargs='?', help="number of training steps", type=int)
    parser.add_argument("-batch", nargs='?', help="size of minibatch", type=int)
    parser.add_argument("-dim", nargs='?', help="dimension of hidden layers", type=int)
    parser.add_argument("-filename", nargs='?', help="filename with example data", type=str)
    parser.add_argument("-limit", nargs='?', help="dummy for compatibility wih lrnns", type=str)  # dummy

    args = parser.parse_args()

    steps = args.ts or 1000
    folds = args.xval or 10
    dim = args.dim or 10
    batch = args.batch or 1

    print(str(args))

    filename = args.filename or "_graphs.pkl"
    dataset_folds = load_dataset_folds_external(args.sd, suffix=filename, batch=batch)

    num_node_features = dataset_folds[0][0].dataset[0].num_node_features

    cross, writer = crossvalidate(args.model.lower(), dataset_folds, args.out, steps, dim=dim)

    content = json.dumps(cross.__dict__, indent=4)

    outp = args.out or "./" + writer.logdir
    with open(outp + "/crossval.json", "w") as f:
        f.writelines(content)
        f.close()
