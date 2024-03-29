#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N inf_onnx
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -l filesystems=eagle:grand:home
#PBS -k doe
#PBS -j oe
#PBS -A datascience
#PBS -q debug-scaling
#PBS -V

DRIVER=inference_onnx.py
MODULE=conda/2022-09-08
CONDA_ENV=base
LOGGING="verbose"

# Set env
cd $PBS_O_WORKDIR

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

# Handle output
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
if [ $LOGGING = "verbose" ] || [ $LOGGING = "verbose-perf" ]; then
    mkdir $JOBID
    mv *.log $JOBID
fi
