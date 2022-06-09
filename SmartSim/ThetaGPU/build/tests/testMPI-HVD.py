import torch
import horovod.torch as hvd
hvd.init()
hrank = hvd.rank()
hsize = hvd.size()
if (hrank == 0):
    print("HVD thread support ", hvd.mpi_threads_supported())
    print("HVD MPI built ",hvd.mpi_built())
    print("HVD MPI enabled ",hvd.mpi_enabled())

import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.threads = False
mpi4py.rc.thread_level = 'multiple'

from mpi4py import MPI 
if not MPI.Is_initialized():
    print("initializing MPI with Init_thread")
    MPI.Init_thread()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if (rank == 0): 
    print("MPI number of threads: ", MPI.Query_thread())
    print("MPI version ", MPI.Get_version())
    print("MPI vendor ", MPI.get_vendor())
    print("MPI library version ", MPI.Get_library_version())

assert hsize == size
assert hrank == rank

hvd.allreduce(torch.tensor(0), name='barrier')
if (hrank == 0):
    print("Past HVD barier")

min_loc = comm.allreduce((1.5*rank,rank), op=MPI.MINLOC)
max_loc = comm.allreduce((1.5*rank, rank), op=MPI.MAXLOC)
if (rank == 0): 
    print("Min val: ",min_loc[0]," at rank: ",min_loc[1])
    print("Max val: ",max_loc[0]," at rank: ",max_loc[1])

