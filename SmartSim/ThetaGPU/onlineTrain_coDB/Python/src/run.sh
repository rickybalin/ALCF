#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim
DRIVER=test.py

module load conda/2021-11-30
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.3.0 smartsim-redis/gcc-9.3.0
export  LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
conda activate $CONDA_ENV

python $DRIVER
