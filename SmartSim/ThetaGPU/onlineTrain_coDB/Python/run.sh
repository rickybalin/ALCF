#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim
DRIVER=src/driver.py

echo nodes $1
echo CPU cores per node $2
echo simprocs $3
echo sim_ppn $4
echo mlprocs $5
echo ml_ppn $6
echo db_ppn $7
echo device $8

module load conda/2021-11-30
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.3.0 smartsim-redis/gcc-9.3.0
export  LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
conda activate $CONDA_ENV
HOST_FILE=$(cat $COBALT_NODEFILE)

python $DRIVER $1 $2 $3 $4 $5 $6 $7 $8 $HOST_FILE
