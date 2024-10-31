#!/bin/bash

cmake \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    -DINTEL_EXTENSION_FOR_PYTORCH_PATH=`python -c 'import torch; print(torch.__path__[0].replace("torch","intel_extension_for_pytorch"))'` \
    ./

make


