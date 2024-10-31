#!/bin/bash

module load frameworks/2024.2.1_u1

# Torch libraries
python -c 'import torch; print(torch.__path__[0])'
python -c 'import torch;print(torch.utils.cmake_prefix_path)'

# IPEX libraries
python -c 'import torch; print(torch.__path__[0].replace("torch","intel_extension_for_pytorch"))'


