from warnings import warn

import dpctl.tensor as dpt
import numpy as np
from dpctl import SyclQueue
from mpi4py import MPI
from sklearn.datasets import load_digits

from sklearnex.spmd.cluster import DBSCAN


def get_data_slice(chunk, count):
    assert chunk < count
    X, y = load_digits(return_X_y=True)
    n_samples, _ = X.shape
    size = n_samples // count
    first = chunk * size
    last = first + size
    return (X[first:last, :], y[first:last])


def get_train_data(rank, size):
    return get_data_slice(rank, size + 1)


def get_test_data(size):
    return get_data_slice(size, size + 1)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X, _ = get_train_data(rank, size)

queue = SyclQueue("gpu")

dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=queue)

model = DBSCAN(eps=3, min_samples=2).fit(dpt_X)

print("")
print(f"Labels on rank {rank} (slice of 2):\n", model.labels_[:2])
print(f"{type(model.labels_)}")
