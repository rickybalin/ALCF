#!/bin/bash
num_gpus=4
offset=0
gpu=$((${PMI_LOCAL_RANK} % ${num_gpus} + ${offset} ))
export CUDA_VISIBLE_DEVICES=$gpu
echo ?RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}?
exec "$@"

