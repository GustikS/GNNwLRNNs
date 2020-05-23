#!/bin/bash
#SBATCH --partition=longjobs
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
ml Java/1.8.0_202 
cd /home/XXXXX/neuralogic/
java -XX:+UseSerialGC -XX:-BackgroundCompilation -XX:NewSize=2000m -Xms2g -Xmx7g -jar /home/XXXXX/neuralogic/NeuraLogic.jar -sd /home/XXXXX/neuralogic/datasets/jair/first10/mol2types/gnnlrnn/CCRF_CEM -t /home/XXXXX/neuralogic/templates/molecules/gnns/gin.txt -ts 2000 -fp fold -init glorot -lr 1.5e-05 -ef XEnt -out /home/XXXXX/neuralogic/experiments/java/results/jair/first10/mol2types/gnnlrnn/CCRF_CEM/molecules_gnns_gin.txt/_-ts_2000_-fp_fold_-init_glorot_-lr_1.5e-05_-ef_XEnt