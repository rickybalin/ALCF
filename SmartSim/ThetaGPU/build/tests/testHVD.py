import numpy as np
import torch
import horovod.torch as hvd
hvd.init()
hrank = hvd.rank()
hsize = hvd.size()
if (hrank == 0):
    print("HVD thread support ", hvd.mpi_threads_supported())
    print("HVD MPI built ",hvd.mpi_built())
    print("HVD MPI enabled ",hvd.mpi_enabled())

hvd.allreduce(torch.tensor(0), name='barrier')
if (hrank == 0):
    print("Past HVD barrier")

arr = np.empty([1,2])
arr[0,0] = -2.0*hrank
arr[0,1] = hrank
tmpTensor = torch.DoubleTensor(arr)
var_array = hvd.allgather(tmpTensor)
var_array = var_array.numpy()
min_val = np.amin(var_array, axis=0)
min_loc = np.argmin(var_array, axis=0)
max_val = np.amax(var_array, axis=0)
max_loc = np.argmax(var_array, axis=0)

assert min_loc[0] == var_array[min_loc[0],1]

if (hrank == 0): 
    print("Min val: ",min_val[0]," at rank: ",min_loc[0])
    print("Max val: ",max_val[0]," at rank: ",max_loc[0])

