#!/bin/bash

START=1
END=256
vers="v5"

# Loop over directories with scaling results
for (( c=$START; c<=$END; c=c*2 ))
do 
     echo "Considering case with $c GPU"
     grep 'Global train loop throughput' $c/* > raw
     awk '{print $5}' raw > "data_${vers}_${c}.dat"
done

# Execute Python prohram to read all data and plot it
python plotScaling.py
