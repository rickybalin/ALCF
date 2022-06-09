#!/bin/bash

runtime=60
nodes=1
queue=single-gpu
filesystem=grand
project=datascience

qsub-gpu -I -A $project -q $queue -n $nodes -t $runtime -attrs filesystems=$filesystem
