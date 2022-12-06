#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N inf_coDB
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l filesystems=eagle:grand:home
#PBS -k doe
#PBS -j oe
#PBS -A cfdml_aesp
#PBS -q debug-scaling
#PBS -V

DRIVER=src/driver_polaris.py
MODULE=conda/2022-09-08
CONDA_ENV=/grand/datascience/balin/Polaris/smartsim_envs/test_my_PR/ssim

# Set env
cd $PBS_O_WORKDIR
module load $MODULE
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

# Run
echo python $DRIVER
python $DRIVER

# Handle output
if [ "$logging" = "verbose" ]; then
    JOB_ID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
    mkdir $JOBID
    mv *.log $JOBID
fi
