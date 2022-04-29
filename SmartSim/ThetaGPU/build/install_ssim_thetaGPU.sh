#!/bin/bash

PREFIX="$1"
ENVNAME=ssim

# NOTE: If you encounter problems cloning repositories or installing packages make sure
# you set the web proxies on thetagpu:
# export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
# export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

# Create conda env
module load conda
conda create --prefix $PREFIX/$ENVNAME python=3.8 -y
conda activate $PREFIX/$ENVNAME

# Load system modules
module use --append /lus/grand/projects/datascience/ashao/local/thetagpu/modulefiles
module load smartsim-deps/gcc-9.3.0 smartsim-redis/gcc-9.3.0

# Clone SmartSim (after next release should be available directly via PyPi)
git clone https://github.com/CrayLabs/SmartSim.git
cd SmartSim

# Install SmartSim with ML backends
pip install -e .[dev,ml]

# Install GPU version python packages
smart build --only_python_packages --device gpu

# Test CPU backends
export SMARTSIM_TEST_DEVICE=cpu
pytest tests/backends
# NOTE: tests/backends/test_onnx.py is known to fail

# Test GPU backends
export SMARTSIM_TEST_DEVICE=gpu
pytest tests/backends
# NOTE: 


