#!/bin/bash

PREFIX="$1"
ENVNAME=ssim


########
### Set up software versions and tags
########

SMARTSIM_REDISAI_TAG=1.2.5
PT_REPO_TAG="master" #"v1.10.0" gives error Submodule 'third_party/eigen' could not be updated
PT_REPO_URL=https://github.com/pytorch/pytorch.git
HOROVOD_REPO_TAG="v0.24.3" # v0.23.0
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
MPI=/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.1_ucx-1.11.2_gcc-9.3.0

CUDA_VERSION_MAJOR=11
CUDA_VERSION_MINOR=4
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_BASE=/usr/local/cuda-$CUDA_VERSION

CUDA_DEPS_BASE=/lus/theta-fs0/software/thetagpu/cuda
CUDNN_VERSION_MAJOR=8
CUDNN_VERSION_MINOR=2
CUDNN_VERSION_EXTRA=4.15 # this is for CUDA 11.4
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn-$CUDA_VERSION-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=11.4-1 # this is for CUDA 11.4
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64

TENSORRT_VERSION_MAJOR=8
TENSORRT_VERSION_MINOR=2.1.8 # this is for CUDA 11.4
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
TENSORRT_BASE=$CUDA_DEPS_BASE/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR



########
### Set up environment
########

export SMARTSIM_REDISAI=$SMARTSIM_REDISAI_TAG

export TF_CUDA_COMPUTE_CAPABILITIES=8.0
export TF_CUDA_VERSION=$CUDA_VERSION_MAJOR
export TF_CUDNN_VERSION=$CUDNN_VERSION_MAJOR
export TF_TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR
export TF_NCCL_VERSION=$NCCL_VERSION_MAJOR
export CUDA_TOOLKIT_PATH=$CUDA_BASE
export CUDNN_INSTALL_PATH=$CUDNN_BASE
export NCCL_INSTALL_PATH=$NCCL_BASE
export TENSORRT_INSTALL_PATH=$TENSORRT_BASE
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_CUDA_CLANG=0
export TF_NEED_OPENCL=0
export TF_NEED_MPI=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=1
export TF_CUDA_PATHS=$CUDA_BASE,$CUDNN_BASE,$NCCL_BASE,$TENSORRT_BASE
export GCC_HOST_COMPILER_PATH=$(which gcc)
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_SET_ANDROID_WORKSPACE=0

export LD_LIBRARY_PATH=$MPI/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_BASE/lib64:$CUDNN_BASE/lib64:$NCCL_BASE/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TENSORRT_BASE/lib:$LD_LIBRARY_PATH
export PATH=$MPI/bin:$PATH
export PATH=$CUDA_BASE/lib64:$CUDNN_BASE/lib64:$NCCL_BASE/lib:$PATH
export PATH=$TENSORRT_BASE/lib:$PATH



########
### Check for outside communication on ThetaGPU
########

# (be sure not to inherit these vars from dotfiles)
unset https_proxy
unset http_proxy
wget -q --spider -T 10 http://google.com
if [ $? -eq 0 ]; then
    echo "Network Online"
else
    # non-/interactive full-node job without --attrs=pubnet on ThetaGPU
    echo "Network Offline, setting proxy envs"
    export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
    export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
fi



########
### Install SmartSim using site modules
########

module load conda/2021-11-30
conda create --prefix $PREFIX/$ENVNAME python=3.8 -y
conda activate $PREFIX/$ENVNAME

module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.3.0 smartsim-redis/gcc-9.3.0

# The smartsim modules prepend path to CUDA 11.3 libraries, but this is not necessarily what I
# set above, so prepend what I actually want
export LD_LIBRARY_PATH=$CUDA_BASE/lib64:$LD_LIBRARY_PATH

git clone https://github.com/CrayLabs/SmartSim.git
cd SmartSim

pip install -e .[dev,ml]
pip install protobuf==3.20.1 # this is a fix because version 4.21.1 installed by deafult gives error with TF

# The command below is recommended, where --only_python_packages means do not build RedisAI
# because it is provided with the module and only build torch (and other future Python packages).
# But this command does not build the correct version of torch needed to use the A100 GPU, it builds
# 1.7.1+cu102.
#smart build --only_python_packages --device gpu
cd ../

# I need a later version of CUDA libraries, 11.1 at least, so I can do what is below
#pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Installing PyTorch is the same as smart build --only_python_packages --device gpu only because
# RedisAI is provided by the module. In general, I need to run smart build!

# PyTorch version torch==1.9.1+cu111 is problematic when also installing Horovod because the CUDNN and NCCL
# dependencies are not all set up fine for CUDA versions 11.1 and 11.3. It seems like for Horovod I need 
# CUDA version 11.4, and this means a custom build of PyTorch, as done below
git clone --recursive $PT_REPO_URL
cd pytorch
if [[ -z "$PT_REPO_TAG" ]]; then
    echo Checkout PyTorch master
else
    echo Checkout PyTorch tag $PT_REPO_TAG
    git checkout --recurse-submodules $PT_REPO_TAG
fi

export CUDA_TOOLKIT_ROOT_DIR=$CUDA_BASE
export NCCL_ROOT_DIR=$NCCL_BASE
export CUDNN_ROOT=$CUDNN_BASE
export USE_TENSORRT=ON
export TENSORRT_ROOT=$TENSORRT_BASE
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#export TENSORRT_LIBRARY=$TENSORRT_BASE/lib/libmyelin.so
export TENSORRT_LIBRARY=$TENSORRT_BASE/lib
export TENSORRT_LIBRARY_INFER=$TENSORRT_BASE/lib/libnvinfer.so
export TENSORRT_LIBRARY_INFER_PLUGIN=$TENSORRT_BASE/lib/libnvinfer_plugin.so
export TENSORRT_INCLUDE_DIR=$TENSORRT_BASE/include

conda install -y pyyaml
python setup.py bdist_wheel
PT_WHEEL=$(find dist/ -name "torch*.whl" -type f)
echo copying pytorch wheel file $PT_WHEEL
cp $PT_WHEEL .
echo pip installing $(basename $PT_WHEEL)
pip install $(basename $PT_WHEEL)

cd ..



########
### Test SmartSim CPU and GPU backends
########

cd SmartSim
export SMARTSIM_TEST_DEVICE=cpu
pytest tests/backends

export SMARTSIM_TEST_DEVICE=gpu
pytest tests/backends
cd ../



########
### Install Horovod
########

echo Clone Horovod
git clone --recursive $HOROVOD_REPO_URL
cd horovod

if [[ -z "$HOROVOD_REPO_TAG" ]]; then
    echo Checkout Horovod master
else
    echo Checkout Horovod tag $HOROVOD_REPO_TAG
    git checkout --recurse-submodules $HOROVOD_REPO_TAG
fi

echo Build Horovod Wheel using MPI from $MPI and NCCL from ${NCCL_BASE}

HOROVOD_CUDA_HOME=${CUDA_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL .
HVD_WHEEL=$(find . -maxdepth 1 -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall $HVD_WHEEL
cd ..



########
## Install MPI4PY
########

env MPICC=$MPI/bin/mpicc pip install mpi4py --no-cache-dir



########
## Install other packages
########

pip install matplotlib


