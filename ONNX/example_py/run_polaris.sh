#!/bin/bash

DRIVER=inference_onnx.py
MODULE=conda/2022-09-08
CONDA_ENV=base

# Set env
module load $MODULE
conda activate $CONDA_ENV
echo Using module $MODULE
echo and conda env $CONDA_ENV
echo

NODES=$(cat $PBS_NODEFILE | wc -l)
GPUS_PER_NODE=1
RANKS=$((NODES * GPUS_PER_NODE))
echo Number of nodes: $NODES
echo Number of ranks per node: $GPUS_PER_NODE
echo Number of total ranks: $RANKS
echo

export MPICH_GPU_SUPPORT_ENABLED=1
HOST_FILE=$(echo $PBS_NODEFILE)

# Run
EXE_ARGS="--model_device cpu --sim_device cpu --ppn ${GPUS_PER_NODE} --logging verbose"
echo mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE python $DRIVER $EXE_ARGS
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE \
 python $DRIVER $EXE_ARGS

