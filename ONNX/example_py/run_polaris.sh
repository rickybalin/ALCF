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
PROCS_PER_NODE=4
RANKS=$((NODES * PROCS_PER_NODE))
echo Number of nodes: $NODES
echo Number of ranks per node: $PROCS_PER_NODE
echo Number of total ranks: $RANKS
echo

export MPICH_GPU_SUPPORT_ENABLED=1
HOST_FILE=$(echo $PBS_NODEFILE)

# Run
EXE_ARGS="--model_device cuda --sim_device cuda --ppn ${PROCS_PER_NODE} --nSamples 2 --logging verbose"
echo mpiexec -n $RANKS --ppn $PROCS_PER_NODE --hostfile $PBS_NODEFILE python $DRIVER $EXE_ARGS
mpiexec -n $RANKS --ppn $PROCS_PER_NODE --hostfile $PBS_NODEFILE \
 python $DRIVER $EXE_ARGS

