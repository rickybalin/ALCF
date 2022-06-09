#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim
FILE=testMPI.py

echo procs $1
echo ppn $2

module load conda/2021-11-30
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.3.0 smartsim-redis/gcc-9.3.0
export  LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
conda activate $CONDA_ENV

echo PATH is 
echo $PATH
echo
echo LD_LIBRARY_PATH is
echo $LD_LIBRARY_PATH
echo

# mpirun options (mpirun -help and mpirun -help mapping)
# -n, --n: number of processes to run
# -c, -np, --np: number of processes to run (same as above)
# -N: launch N processes per node on all allocated nodes
# -oversubsrribe, --oversubscribe: nodes are allowed to be oversubscribed
# -H, -host, --host: list of hosts to invoke processes on
# -hostfile: Cobalt nodelfile created by job ($COBALT_NODEFILE)

#HOSTS=$(cat $COBALT_NODEFILE | sed ':a;N;$!ba;s/\n/,/g')
#mpirun -host $HOSTS -oversubscribe -n $1 -N $2 python $FILE
mpirun -hostfile $COBALT_NODEFILE -n $1 -N $2 -x LD_LIBRARY_PATH -x PATH python $FILE
