#!/bin/bash

module load  frameworks/2023.12.15.001
export ZE_AFFINITY_MASK=0.0 # needed to prevent pt_read_dpnp/dpctl error

