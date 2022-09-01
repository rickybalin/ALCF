# Scaling Tests

This folder contains the scripts used to perform scaling tests on the GPU and CPU of various machines at ALCF. 
There are scripts for three different NN models of different sizes and types and there are scripts to test both Horovod and DDP. 
There are also some bash and plotting scripts to visualize the results of the tests. 
The QoI is the scaling efficiency, which is defined below
```
scaling efficiency = (parallel throughput) / (serial throughput x num. ranks)
```
