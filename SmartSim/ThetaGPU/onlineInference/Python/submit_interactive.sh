#!/bin/bash

# change `datascience` to the charge account for your project
# change `filesystem` to the filesystem of your working directory
ChargeAccount=datascience
queue=full-node
runtime=60
filesystem=theta-fs0

# args:
dbnodes=1
simnodes=1
nodes=$(($dbnodes + $simnodes))
ppn=48 # CPU cores per node
simprocs=2

echo number of total nodes $nodes 
echo time in minutes $runtime
echo number of total processes $allprocs
echo ppn  N $ppn
echo queue $queue
 
qsub-gpu -M riccardo.balin@colorado.edu -I -q $queue -n $nodes -t $runtime -A $ChargeAccount --attrs filesystems=$filesystem

# Then, run
#./run.sh 48 2 1 1 2
