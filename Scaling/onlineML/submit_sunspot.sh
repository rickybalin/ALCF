#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N anisoSGS_ddp_1
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -k doe
#PBS -j oe
#PBS -A cfdml_aesp_CNDA
##PBS -A Aurora_deployment
#PBS -q workq
#PBS -V

EXE=./src/train_driver.py

module load frameworks/2022.12.30.001
echo Using frameworks 2022.12.30.001
echo 

export ZE_AFFINITY_MASK=0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1,4.0,4.1,5.0,5.1
export EnableImplicitScaling=0
export IPEX_TILE_AS_DEVICE=1
export IPEX_XPU_ONEDNN_LAYOUT_OPT=1
#export CCL_LOG_LEVEL=INFO
export CCL_ZE_QUEUE_INDEX_OFFSET=0
export CCL_SYCL_OUTPUT_EVENT=0
export CCL_OP_SYNC=1
#export HOROVOD_LOG_LEVEL=INFO
export HOROVOD_CCL_FIN_THREADS=1
export HOROVOD_CCL_ADD_EXTRA_WAIT=1
export HOROVOD_FUSION_THRESHOLD=150000000
export HOROVOD_CYCLE_TIME=0.1

cd $PBS_O_WORKDIR

NODES=$(cat $PBS_NODEFILE | wc -l)
GPUS_PER_NODE=1
RANKS=$((NODES * GPUS_PER_NODE))
BIND_LIST="0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"
#NDEPTH=$(( 208/$GPUS_PER_NODE ))
echo Number of nodes: $NODES
echo Number of ranks per node: $GPUS_PER_NODE
echo Number of total ranks: $RANKS
#echo Binding depth: $NDEPTH
echo

echo mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE python $EXE
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE \
        -l --cpu-bind=verbose,list:$BIND_LIST \
        ./affinity_ml.sh \
        python $EXE

#--cpu-bind depth --depth ${NDEPTH}


