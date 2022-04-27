# Build a Conda Env with SmartSim and SmartRedis Modules on Theta

1. Create a new conda env and install SmartSim and SmartRedis with `source install_ssim_theta.sh /path/to/env`. It is recommended to specify a path to a preject space, not a user's home space.
2. Install Horovod with `./install_horovod_theta.sh`. Note: this should be run right after the installation of SmartSim on the same terminal. Otherwise, the terminal environment needs to be set as specified at the top of `install_ssim_theta.sh`.
3. Install KeyDB with `./install_keyDB.sh`. Instructions on how to use KeyDB can be found at the [SmartSim documentation](https://www.craylabs.org/docs/orchestrator.html#keydb).
4. Install mpi4py with `./install_mpi4py.sh`. This will be needed by some of the online inference examples. NOTE: currently there is a know issue with using Horovod and mpi4py together, where mpi4py crashes on collective comms with more than 16 mpi processes. I think the issue with threading support mismatch between Horovod and mpi4py and it is being investigated.
