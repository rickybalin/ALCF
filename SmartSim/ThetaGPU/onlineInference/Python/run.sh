#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV_PREFIX=/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim
DRIVER=src/driver.py

echo ppn $1
echo nodes $2
echo dbnodes $3
echo simnodes $4
echo simprocs $5

module load conda
conda activate $CONDA_ENV_PREFIX

python $DRIVER $1 $2 $3 $4 $5
