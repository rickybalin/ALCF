#!/bin/bash

# change `datascience` to the charge account for your project
# change `filesystem` to the filesystem of your working directory
ChargeAccount=datascience
queue=single-gpu
runtime=60
filesystem=theta-fs0,grand

# args:
nodes=1
ppn=48 # CPU cores per node
simprocs=8
sim_ppn=8
db_ppn=4

echo number of nodes $nodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo number of db processes per node $db_ppn
echo CPU cores per node $ppn
echo run time in minutes $runtime
 
qsub -I -q $queue -n $nodes -t $runtime -A $ChargeAccount --attrs filesystems=$filesystem

# Then, from the MOM nodes, run
#./run.sh 2 48 2 2 2
