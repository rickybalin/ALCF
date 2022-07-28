#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV_PREFIX=/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim
DRIVER=src/driver.py

echo nodes $1
echo ppn $2
echo simprocs $3
echo sim_ppn $4
echo db_ppn $5

module load conda
conda activate $CONDA_ENV_PREFIX
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.3.0 smartsim-redis/gcc-9.3.0

python $DRIVER $1 $2 $3 $4 $5
