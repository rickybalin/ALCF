#!/bin/bash

# change `datascience` to the charge account for your project
# change `filesystem` to the filesystem of your working directory
ChargeAccount=datascience
queue=full-node
runtime=120
filesystem=theta-fs0,grand

# args:
nodes=2
ppn=128 # CPU cores per node
simprocs=128
sim_ppn=64 # CPU cores per node assigned to sim
mlprocs=8
ml_ppn=8 # CPU cores per node assigned to ML
db_ppn=32 # CPU cores per node assigned to DB
device=cpu

echo number of nodes $nodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo number of ML processes $mlprocs
echo number of ML processes per node $ml_ppn
echo number of db processes per node $db_ppn
echo CPU cores per node $ppn
echo run time in minutes $runtime

qsub -I -q $queue -n $nodes -t $runtime -A $ChargeAccount --attrs filesystems=$filesystem

# Then, run
#./run.sh 1 12 4 4 1 1 4 cpu
