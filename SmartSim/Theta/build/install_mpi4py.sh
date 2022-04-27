#!/bin/bash

### Install mpi4py

# NOTE: - load desired ssim conda environment and other Theta modules that go with it

export CC=$(which cc)
export MPICC=$(which cc)

python -m pip install --upgrade pip
env MPICC=$(which cc) python -m pip install mpi4py

