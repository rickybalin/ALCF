#!/bin/bash

CONDA_ENV=/grand/datascience/balin/Polaris/smartsim_envs/test_my_PR/ssim
DRIVER=src/driver_polaris.py

module load conda/2022-09-08
conda activate $CONDA_ENV
export MPICH_GPU_SUPPORT_ENABLED=1

python $DRIVER
