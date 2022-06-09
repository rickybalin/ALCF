#!/bin/bash

# change `datascience` to the charge account for your project
# change `filesystem` to the filesystem of your working directory
ChargeAccount=datascience
queue=full-node
runtime=30
filesystem=theta-fs0,grand

# args:
nodes=2
ppn=48 # CPU cores per node
simprocs=48
sim_ppn=24
db_ppn=24

echo number of nodes $nodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo number of db processes per node $db_ppn
echo CPU cores per node $ppn
echo run time in minutes $runtime


qsub -q $queue -n $nodes -t $runtime -A $ChargeAccount --attrs filesystems=$filesystem run.sh $nodes $ppn $simprocs $sim_ppn $db_ppn

