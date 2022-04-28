#!/bin/bash

CC=cc
CXX=CC
FC=ftn

cmake \
-DCMAKE_Fortran_FLAGS="-craype-verbose -g" \
-DCMAKE_CXX_FLAGS="-craype-verbose -g" \
-DCMAKE_C_FLAGS="-craype-verbose -g" \
-DSSIMLIB=/projects/cfdml_aesp/balin/SmartSim/smartredis-0.3.0 \
./

make
