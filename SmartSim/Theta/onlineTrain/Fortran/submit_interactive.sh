#!/bin/bash

# change `datascience` to the charge account for your project
# change `filesystem` to the filesystem of your working directory
ChargeAccount=datascience
queue=debug-cache-quad
runtime=60
filesystem=theta-fs0

# args:
dbnodes=1
simnodes=2
mlnodes=1
nodes=$(($dbnodes + $simnodes + $mlnodes))
ppn=64 # cores per node
simprocs=128
mlprocs=64

echo number of total nodes $nodes 
echo time in minutes $runtime
echo number of simulation processes $simprocs
echo ppn  N $ppn
echo queue $queue
 
qsub -I -q $queue -n $nodes -t $runtime -A $ChargeAccount --attrs filesystems=$filesystem

# Then, from the MOM nodes, run
#./run.sh 64 4 1 2 1 128 64
