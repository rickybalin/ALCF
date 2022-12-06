#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/projects/cfdml_aesp/balin/SmartSim_thetaGPU_v0.4.1/ssim
DRIVER=src/driver_thetagpu.py

module load conda/2022-07-01
module load openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.4.0/v0.4.1 
#module load smartsim-redis/gcc-9.4.0/v0.4.1
export  LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
conda activate $CONDA_ENV
HOST_FILE=$(cat $COBALT_NODEFILE)

python $DRIVER
