**Prepare Data

1) go to ftp://ftp.ics.uci.edu/pub/baldig/learning/nci/gi50/ and download and UNZIP all the files into some folder DIR
2) run "java -jar mol2csv.jar DIR" 
	- this will convert all the molecular datasets in the DIR into simple csv representation, creating one folder per each dataset
	- the source code of this simple utility can be found at: https://github.com/GustikS/NeuraLogic/blob/master/Resources/src/main/java/cz/cvut/fel/ida/utils/molecules/preprocessing/Molecule2csv.java
3) run "python csvs2graphs.py DIR OUTDIR"
	- this will transform the csv representations into respective graph objects (for PyG and DGL) and prolog representation (for LRNN) and also split into 10 train-val-test folds
		- we note that as we use only the types in this case, and not full feature vectors, this creates unnecessarily large files/graph with many zeros, as we do not treat this as a special case (with sparse embedding indices instead of dense vectors). This is of course terribly inefficient, but has no influence on the main point of the experiments (all the frameworks use the same representation).

**Run Experiments

1) run the PyG script with some model (gcn,gsage,gin) on some of the processed datasets, e.g. for the first one (786_0):
	"python run_script_pyg.py -sd DIR/786_0 -ts 2000 -model gin -lr 1.5e-05 -ts 2000 -out OUTDIR/pyg/786_0"
2) similarly, run the DGL
	"python run_script_dgl.py -sd DIR/786_0 -ts 2000 -model gin -lr 1.5e-05 -ts 2000 -out OUTDIR/dgl/786_0"
3) run LRNN on the same data and models by calling:
	"java -jar NeuraLogic.jar -sd DIR/786_0 -t ./templates/gin.txt -ts 2000 -fp fold -init glorot -lr 1.5e-05 -ef XEnt -out OUTDIR/lrnn/786_0"

**Analyze Results

you can analyze the results manually in the respective folders, here is a script to do some loading and plotting used for the paper:
