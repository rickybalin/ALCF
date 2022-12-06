#!/bin/bash -l
#COBALT -A datascience
#COBALT -q full-node
#COBALT -t 30
#COBALT --attrs filesystems=theta-fs0,eagle
#COBALT -n 2

DRIVER=src/driver_thetagpu.py
MODULE=conda/2022-07-01
CONDA_ENV=/projects/cfdml_aesp/balin/SmartSim_thetaGPU_v0.4.1/ssim

# Set env
module load $MODULE
module load openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.4.0/v0.4.1 
#module load smartsim-redis/gcc-9.4.0/v0.4.1
export  LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
conda activate $CONDA_ENV
HOST_FILE=$(echo $COBALT_NODEFILE)

# Run
echo python $DRIVER
python $DRIVER

# Handle output
if [ "$logging" = "verbose" ]; then
    mkdir $COBALT_JOBID
    mv *.log $COBALT_JOBID
fi
