#!/bin/bash

module use /soft/modulefiles
module load conda/2024-04-29
conda activate

# Torch libraries
python -c 'import torch; print(torch.__path__[0])'
python -c 'import torch;print(torch.utils.cmake_prefix_path)'



