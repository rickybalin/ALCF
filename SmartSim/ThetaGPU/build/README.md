# Build a New Conda Env for SmartSim on ThetaGPU

1. Submit an interactive job to ThetaGPU grabbing one GPU with `./subInteractive_ThetaGPU.sh`. Remember to change the project name from *datascience* to your specific project.
2. Create a new conda env and install SmartSim and SmartRedis by executing
```
source install_ssim_thetaGPU.sh /path/to/conda/env
```
Note that:
- It is recommended to build the env in a project space rather than in a user's home space.
- The backend test `tests/backends/test_onnx.py` is known to fail on both the CPU and GPU
- The backend test `tests/backends/test_torch.py` often fails on the GPU. This is a known issue, caused by the previous tensorflow test which does not release GPU memory in time and so triggers an out of memory error. Simply run the single test again with `pytest tests/backends/test_torch.py` and it will pass.
