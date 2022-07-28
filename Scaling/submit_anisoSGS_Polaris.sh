#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N anisoSGS_scaling
#PBS -l walltime=00:30:00
#PBS -l select=4:ncpus=64:ngpus=4
#PBS -k doe
#PBS -j oe
#PBS -A datascience

cd $PBS_O_WORKDIR

EXE=scaling_anisoSGS_pt_hvd.py
NODES=$(cat $PBS_NODEFILE | wc -l)
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo Number of nodes: $NODES
echo Number of ranks per node: $GPUS_PER_NODE
echo Number of total ranks: $RANKS

module load conda/2022-07-19
conda activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE \
 python $EXE --device cuda
