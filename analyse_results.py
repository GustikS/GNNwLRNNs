from grid.loading_results  import *

#%% analyze PyG

experiment_path = "./original_results/pyg"

data_pyg = analyse(experiment_path, plot=False, metrics=["python"])

#%% analyze DGL

experiment_path = "./original_results/dgl"

data_dgl = analyse(experiment_path, plot=False, metrics=["python"])

#%% analyze LRNN

experiment_path = "./original_results/lrnn"

data_lrnn = analyse(experiment_path, plot=False, metrics=convert.keys())
