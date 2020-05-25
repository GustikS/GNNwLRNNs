###### Prerequisites

For the GNN frameworks, please follow the instructions at their respective pages, i.e. [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and [DGL](https://www.dgl.ai/pages/start.html). They are both very nice and user friendly python frameworks, so it should go smoothly. 
For reference, we used PyG 1.4.3 and DGL 0.4.3 (actual versions as of March 2020).
Additionally, you will also need some basic python stuff like Pandas and Matplotlib to analyse the results, but you probably already have those anyway.

For the LRNN framework, all you need is Java >= 1.8. The whole framework is the small `NeuraLogic.jar` included in this repo (with source at [LRNN](https://github.com/GustikS/NeuraLogic))

##### Download & Create the Datasets

run the script to download and process all (73) the NCI datasets. 

`bash create_datasets.sh`

* Note this will take a considerable disk space (few GBs), so you might want to limit yourself, e.g. to the first 10 datasets as in the paper. For that follow the individual steps below and remove the unwanted datasets (e.g. after donwload).


###### Alternative: individual steps - choose only the datasets you want:

1. go to ftp://ftp.ics.uci.edu/pub/baldig/learning/nci/gi50/ and download and UNZIP all the files into some folder DIR
1. run `java -jar mol2csv.jar DIR`
	- this will convert all the molecular datasets in the DIR into simple csv representation, creating one folder per each dataset
	- the source code of this simple utility can be found at: [Molecule2csv](https://github.com/GustikS/NeuraLogic/blob/master/Resources/src/main/java/cz/cvut/fel/ida/utils/molecules/preprocessing/Molecule2csv.java)
1. run `python csvs2graphs.py DIR OUTDIR`
	- this will transform the csv representations into respective graph objects (for PyG and DGL) and textual (datalog) representation (for LRNN), and also split into 10 train-val-test folds
		- we note that as we use only the mol2 types in this case, and not the full feature vectors, this creates unnecessarily large files/graph with many zeros, as we do not treat this as a special case (with sparse embedding indices instead of dense vectors). This is of course terribly space-inefficient, but has no influence on the main point of the experiments (all the frameworks use the same representation). Actually it causes LRNN to consume much more memory than it should due to complex parsing of the bloated text files, so we might fix this in near future.

##### Run the Experiments

1. run the PyG script with some model (gcn/gsage/gin) on some of the processed datasets, e.g. for the first dataset (NCI 786_0) as:

	`python run_script_pyg.py -sd DIR/786_0 -model gcn -lr 1.5e-05 -ts 2000 -out OUTDIR/pyg/786_0`
1. very similarly, run the DGL version by:

	`python run_script_dgl.py -sd DIR/786_0 -model gcn -lr 1.5e-05 -ts 2000 -out OUTDIR/dgl/786_0`
	
1. run the LRNN framework on the same datasets and models (templates) by calling:

	`java -Xmx5g -jar NeuraLogic.jar -sd DIR/786_0 -t ./templates/gcn.txt -ts 2000 -fp fold -init glorot -lr 1.5e-05 -ef XEnt -out OUTDIR/lrnn/786_0`

1. Change the parameters of the scripts as you like (models, datasets, batch sizes, training steps, learning rates, ...) to further compare the behavior and runtimes of the frameworks, as done in the additional experiments in the paper.

  * We do not recommend to run the full batch of experiments on your local computer - it will take a looong time! Rather, run them on some cluster to parallelize over individual instances. For reference, we include our batch job scripts to run with a Slurm cluster manager in the directory: `./grid`

##### Analyze Results

All the relevant information from the experiments gets stored in the JSON files in the respective OUTDIR directories. For reference, we include our original results from experiments run with the included batch job scripts.

You can analyze our and your new results by own means in the respective JSON files, but we also include a convenience script to do some loading into DataFrames and plotting of the results (used for the paper), so you might want to bootstrap from there:

`./analyse_results.py`

Please let us know if you find any bugs, misconceptions, or anything interesting!


You can find the Lifted Relational Neural Networks framework itself being developed here: https://github.com/GustikS/NeuraLogic