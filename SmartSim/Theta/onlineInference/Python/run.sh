#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV_PREFIX=/projects/cfdml_aesp/balin/SmartSim/ssim
DRIVER=src/driver.py

module swap PrgEnv-intel PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic

echo ppn $1
echo nodes $2
echo dbnodes $3
echo simnodes $4
echo simprocs $5

module load miniconda-3/2021-07-28
conda activate $CONDA_ENV_PREFIX

python $DRIVER $1 $2 $3 $4 $5
