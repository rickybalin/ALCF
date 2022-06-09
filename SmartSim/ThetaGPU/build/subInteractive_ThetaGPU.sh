#!/bin/bash

nodes=1
runtime=120
queue=full-node
project=datascience
filesystem=theta-fs0,grand


qsub-gpu -I -n $nodes -t $runtime -q $queue -A $project --attrs filesystems=$filesystem
