#!/bin/bash

# If correct env from SmartSim installation not already loaded
#module swap PrgEnv-intel PrgEnv-gnu
#export CRAYPE_LINK_TYPE=dynamic
#module load miniconda-3/2021-07-28

# Clone and build KeyDB 
git clone https://www.github.com/eq-alpha/keydb.git --branch v6.2.0
cd keydb/
CC=gcc CXX=g++ make -j 8


########################################################
# NOTES:
# Copy the server and configuration files over to smartsim and replace them with the Redis files. SmartSim will then be tricked into running KeyDB instead of Redis

# cp keydb/src/keydb-server smartsim/smartsim/bin/
# cp keydb/keydb.conf smartsim/smartsim/database/
# cd smartsim/smartsim/bin/
# mv redis-server redis-server.orig
# ln -s keydb-server redis-server
# cd ../database/
# mv redis6.conf redis6.conf.orig
# ln -s keydb.conf redis6.conf

# Change the keydb.conf configuration file as follows 
# loglevel: change to verbose if debugging, otherwise notice or warning as verbose writing slows performance
# save: comment out all save options and replace with empty string ""
# maxclients: 25000
# appendfsync: comment all options out
# server-threads: 8
# bind: comment out
# protected-mode: no

# Check to see if keyDB server works
# ../bin/redis-server ./redis6.conf


