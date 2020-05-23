##### Download & Create the Datasets

run the script to create all (73) datasets. Note this will take a considerable disk space (few GBs), so you might want to limit yourself, e.g. to the first 10 datasets as in the paper. Follow the individual steps for that and remove the unwanted datasets (e.g. after donwload).

`bash create_datasets.sh`

###### Alternative: individual steps - choose only the datasets you want:

1. go to ftp://ftp.ics.uci.edu/pub/baldig/learning/nci/gi50/ and download and UNZIP all the files into some folder DIR
1. run `java -jar mol2csv.jar DIR`
	- this will convert all the molecular datasets in the DIR into simple csv representation, creating one folder per each dataset
	- the source code of this simple utility can be found at: [Molecule2csv](https://github.com/GustikS/NeuraLogic/blob/master/Resources/src/main/java/cz/cvut/fel/ida/utils/molecules/preprocessing/Molecule2csv.java)
1. run `python csvs2graphs.py DIR OUTDIR`
	- this will transform the csv representations into respective graph objects (for PyG and DGL) and prolog representation (for LRNN) and also split into 10 train-val-test folds
		- we note that as we use only the types in this case, and not full feature vectors, this creates unnecessarily large files/graph with many zeros, as we do not treat this as a special case (with sparse embedding indices instead of dense vectors). This is of course terribly inefficient, but has no influence on the main point of the experiments (all the frameworks use the same representation).

##### Run the Experiments

1. run the PyG script with some model (gcn,gsage,gin) on some of the processed datasets, e.g. for the first dataset (NCI 786_0) as:

	`python run_script_pyg.py -sd DIR/786_0 -model gcn -lr 1.5e-05 -ts 2000 -out OUTDIR/pyg/786_0`
1. very similarly, run the DGL version by:

	`python run_script_dgl.py -sd DIR/786_0 -model gcn -lr 1.5e-05 -ts 2000 -out OUTDIR/dgl/786_0`
	
1. run LRNN framework on the same datasets and models by calling:

	`java -Xmx4g -jar NeuraLogic.jar -sd DIR/786_0 -t ./templates/gcn.txt -ts 2000 -fp fold -init glorot -lr 1.5e-05 -ef XEnt -out OUTDIR/lrnn/786_0`

1. Change the parameters of the scripts as you like (models, datasets, batch sizes, training steps, ...) to further compare the behavior and runtimes of the frameworks, as done in the additional experiments in the paper.

* We do not recommend to run the full batch of experiments on your local computer - it will take a looong time! Rather, run them on some cluster to parallelize over individual instances. For reference, we include our batch job scripts to run with a Slurm cluster manager in the directory: `./grid`

##### Analyze Results

All the relevant information from the experiments gets stored in JSON files in the respective OUTDIR directories. For reference, we include our original results from experiments run with the included batch job scripts.

You can analyze our and your new results by own means in the respective JSON files, but we also include a convenience script to do some loading into DataFrames and plotting of the results (used for the paper), so you might want to bootstrap from there:

`./analyse_results.py`

Please let us know if you find any bugs, misconceptions, or anything interesting!


You can find the Lifted Relational Neural Networks framework itself being developed here: https://github.com/GustikS/NeuraLogic