# Build a New Conda Env for SmartSim on ThetaGPU

1. Submit an interactive job to ThetaGPU grabbing a full node with `./subInteractive_ThetaGPU.sh`. Remember to change the project name from *datascience* to your specific project and select all the filesystems you need.
2. Create a new conda env and install SmartSim, SmartRedis, PyTorch, Horovod and mpi4py by executing
```
source install_ssim_hvd_mpi4py_thetaGPU.sh /path/to/conda/env
```
3. Test that the build is correct by running the scripts inside of the `tests/` directory. Use the `run.sh` script and select which test to run with the variable `FILE`. There are a few mpi4py and Horovod tests, as well as a serial and data parallel training example which can run on the CPU or GPU.

Note that:
- It is recommended to build the env in a project space rather than in a user's home space.
- The backend test `tests/backends/test_torch.py` often fails on the GPU. This is a known issue, caused by the previous tensorflow test which does not release GPU memory in time and so triggers an out of memory error. Simply run the single test again with `pytest tests/backends/test_torch.py` and it will pass.
- RedisAI version 1.2.5 is needed to build the correct backends for SmartSim
- Attention needs to paid to the versions of CUDA, CUDNN, NCCL and TensorRT libraries. Not all versions are available on ThetaGPU, and one must pick a consistent set of libraries as was done in the script.
- The site modules provided by SmartSim provide a build of RedisAI, meaning that the script does not have to build that component. This results in the command
```
smart build --only_python_packages --device gpu
```
to be run instead of the usual
```
smart build --device gpu
```
The `--only_python_packages` indicates that RedisAI does not need to be build, but only the Python packages should be built. Currently, this is only PyTorch, but it will be more in the future.
- The installation of PyTorch also requires some attention. The command above with RedisAI version 1.2.5 exported to the environment builds torch 1.7.1+cu102. This version, however, is not enabled to run on the A100 GPU. We need CUDA libraries of version 11.0 or above. One could run
```
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
or similar, for instance, and use one of the torch and CUDA combinations available at the link. However, due to the limited versions of CUDNN and NCCL on ThetaGPU, none of the options available would be compatible with the Horovod build. The solution is to make a custom build of PyTorch with the correct CUDA libraries, in this case 11.4.
- If there are ny other packages that one needs in this environment, they can be added at the bottom of the script.
