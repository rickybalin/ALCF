#!/bin/bash -l
#COBALT -A datascience
#COBALT -q full-node
#COBALT -t 10
#COBALT --attrs filesystems=theta-fs0,grand
#COBALT -n 2

# args:
CONDA_ENV=base
#CONDA_ENV=/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim
FILE=/lus/grand/projects/datascience/balin/ThetaGPU/scaling/ResNet/pytorch_hvd_hacked_imageNet.py
MODEL=resnet152
NAME=test
nodes=2
procs=16
ppn=8 # processes per node

echo number of nodes $nodes
echo number of processes $procs
echo number of processes per node $ppn
echo conda environment $CONDA_ENV

# Set env
module load conda/2022-07-01
conda activate $CONDA_ENV

# Run
echo Executing ...
mpirun -hostfile $COBALT_NODEFILE -n $procs -npernode $ppn -x LD_LIBRARY_PATH -x PATH \
 python $FILE --epoch 40 --arch $MODEL --name $NAME --nnodes $procs

