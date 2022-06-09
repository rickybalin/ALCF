#!/bin/bash

ENV_PATH=$(pwd)

# Load base env
module load conda/2021-09-22
conda activate

# Clone to projet dir
conda create --prefix $ENV_PATH/nept --clone base
conda activate $ENV_PATH/nept

# Install Neptune client
pip install neptune-client
