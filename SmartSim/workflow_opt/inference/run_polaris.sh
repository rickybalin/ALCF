#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/grand/datascience/balin/Polaris/smartsim_envs/test_my_PR/ssim
#CONDA_ENV=/grand/projects/datascience/balin/Polaris/smartsim_envs/buildFromClone/ssim
DRIVER=src/driver_polaris.py

module load conda/2022-09-08
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/grand/projects/datascience/balin/Polaris/smartsim_envs/buildFromClone/ssim/lib


python $DRIVER
