#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N anisoSGS_scaling
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l filesystems=home:grand
#PBS -k doe
#PBS -j oe
#PBS -A datascience
#PBS -q debug-scaling
#PBS -V

EXE=scaling_anisoSGS_pt_hvd.py
MODULE=conda/2022-09-08
ENV=base

module load $MODULE 
conda activate $ENV
echo Using module $MODULE
echo and conda env $ENV
echo 

cd $PBS_O_WORKDIR

NODES=$(cat $PBS_NODEFILE | wc -l)
CPUS_PER_NODE=8
RANKS=$((NODES * CPUS_PER_NODE))
NDEPTH=$(( 32/$CPUS_PER_NODE ))
echo Number of nodes: $NODES
echo Number of ranks per node: $CPUS_PER_NODE
echo Number of total ranks: $RANKS
echo Binding depth: $NDEPTH
echo

echo mpiexec -n $RANKS --ppn $CPUS_PER_NODE --hostfile $PBS_NODEFILE python $EXE --device cpu
mpiexec -n $RANKS --ppn $CPUS_PER_NODE --hostfile $PBS_NODEFILE \
 --cpu-bind depth --depth ${NDEPTH} \
 python $EXE --device cpu --nEpochs 40 --batch 512 --nSamples 32000 \
 --precision fp32
