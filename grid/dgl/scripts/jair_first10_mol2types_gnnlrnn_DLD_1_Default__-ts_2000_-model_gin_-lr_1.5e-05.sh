#!/bin/bash
#SBATCH --partition=longjobs
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
cd /home/XXXXX/neuralogic/
ml Anaconda3
source /mnt/appl/software/Anaconda3/2019.07/etc/profile.d/conda.sh
conda activate /home/XXXXX/neuralogic/anaconda/pytorch14
python3 run_script_dgl.py -sd /home/XXXXX/neuralogic/datasets/jair/first10/mol2types/gnnlrnn/DLD_1 -ts 2000 -model gin -lr 1.5e-05 -out /home/XXXXX/neuralogic/experiments/dgl/results/jair/first10/mol2types/gnnlrnn/DLD_1/Default/_-ts_2000_-model_gin_-lr_1.5e-05