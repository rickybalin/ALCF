#!/bin/bash

# change `datascience` to the charge account for your project
# change `filesystem` to the filesystem of your working directory
ChargeAccount=datascience
queue=debug-cache-quad
runtime=30
filesystem=theta-fs0

# args:
nodes=2
ppn=64 # cores per node
simprocs=64
sim_ppn=32
db_ppn=32

echo number of nodes $nodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo number of db processes per node $db_ppn
echo cores per node $ppn
echo run time in minutes $runtime


qsub -q $queue -n $nodes -t $runtime -A $ChargeAccount --attrs filesystems=$filesystem run.sh $nodes $ppn $simprocs $sim_ppn $db_ppn

