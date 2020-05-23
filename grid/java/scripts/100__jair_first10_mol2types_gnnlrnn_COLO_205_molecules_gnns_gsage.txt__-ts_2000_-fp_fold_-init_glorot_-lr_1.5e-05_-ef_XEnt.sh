#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
ml Java/1.8.0_202 
cd /home/XXXXX/neuralogic/
java -XX:+UseSerialGC -XX:-BackgroundCompilation -XX:NewSize=2000m -Xms2g -Xmx19g -jar /home/XXXXX/neuralogic/NeuraLogic.jar -sd /home/XXXXX/neuralogic/datasets/jair/first10/mol2types/gnnlrnn/COLO_205 -t /home/XXXXX/neuralogic/templates/molecules/gnns/gsage.txt -ts 2000 -fp fold -init glorot -lr 1.5e-05 -ef XEnt -ts 10 -limit 10  -out /home/XXXXX/neuralogic/experiments/java/results/jair/first10/mol2types/gnnlrnn/COLO_205/molecules_gnns_gsage.txt/_-ts_2000_-fp_fold_-init_glorot_-lr_1.5e-05_-ef_XEnt_dummy