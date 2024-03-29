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
LOGGING="verbose-perf"

# Set env
cd $PBS_O_WORKDIR
module load $MODULE
conda activate $CONDA_ENV
export MPICH_GPU_SUPPORT_ENABLED=1
HOST_FILE=$(echo $PBS_NODEFILE)

# Run
echo python $DRIVER
python $DRIVER

# Handle output
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
if [ $LOGGING = "verbose" ] || [ $LOGGING = "verbose-perf" ]; then
    mkdir $JOBID
    mv *.log $JOBID
    mv inference.* $JOBID
fi
