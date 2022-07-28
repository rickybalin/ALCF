#!/bin/bash -l
#COBALT -A datascience
#COBALT -q bigmem
#COBALT -t 30
#COBALT --attrs filesystems=theta-fs0,grand
#COBALT -n 2

# args:
CONDA_ENV=/projects/cfdml_aesp/balin/SmartSim_thetaGPU/ssim
DRIVER=src/driver.py
nodes=2
ppn=64 # CPU cores per node
simprocs=64
sim_ppn=32 # CPU cores per node assigned to sim
mlprocs=16
ml_ppn=8 # CPU cores per node assigned to ML
db_ppn=16 # CPU cores per node assigned to DB
device=cuda

echo number of nodes $nodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo number of ML processes $mlprocs
echo number of ML processes per node $ml_ppn
echo number of db processes per node $db_ppn
echo CPU cores per node $ppn
echo conda environment $CONDA_ENV

# Set env
source /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3/setup.sh
module load conda/2021-11-30
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.3.0 smartsim-redis/gcc-9.3.0
export  LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
conda activate $CONDA_ENV
HOST_FILE=$(echo $COBALT_NODEFILE)

# Run
echo python $DRIVER $nodes $ppn $simprocs $sim_ppn $mlprocs $ml_ppn $db_ppn $device $HOST_FILE
python $DRIVER $nodes $ppn $simprocs $sim_ppn $mlprocs $ml_ppn $db_ppn $device $HOST_FILE
