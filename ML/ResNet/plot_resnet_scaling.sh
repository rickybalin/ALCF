#!/bin/bash

START=1
END=128

# Loop over directories with scaling results
for (( c=$START; c<=$END; c=c*2 ))
do 
     echo "Considering case with $c GPU"
     grep throughput $c/*.output > raw
     awk '{print $3}' raw > data_$c.dat
done

# Execute Python prohram to read all data and plot it
python plotScaling.py
