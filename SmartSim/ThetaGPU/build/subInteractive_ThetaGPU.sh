#!/bin/bash

nodes=1
runtime=60
queue=single-gpu
project=datascience
filesystem=theta-fs0


qsub-gpu -I -n $nodes -t $runtime -q $queue -A $project --attrs filesystems=$filesystem
