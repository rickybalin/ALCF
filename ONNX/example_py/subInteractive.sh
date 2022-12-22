#!/bin/bash

runtime=01:00:00
project=datascience
queue=preemptable

qsub -I -l select=1:ncpus=64:ngpus=4,walltime=$runtime,filesystems=home:grand -q $queue -A $project
