import dpctl.tensor as dpt
import numpy as np
from dpctl import SyclQueue
from time import perf_counter
from typing import NamedTuple
#from mpi4py import MPI

#from sklearnex.spmd.cluster import DBSCAN
from sklearnex import patch_sklearn, config_context
patch_sklearn()
from sklearnex.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

SEED = 7777777
DEFAULT_EPS = 0.6
DEFAULT_MINPTS = 20

class DataSize(NamedTuple):
    n_samples: int
    n_features: int

class Params(NamedTuple):
    eps: float
    minpts: int

OPTIMAL_PARAMS = {
    DataSize(n_samples=2**8, n_features=2): Params(eps=0.173, minpts=4),
    DataSize(n_samples=2**8, n_features=3): Params(eps=0.35, minpts=6),
    DataSize(n_samples=2**8, n_features=10): Params(eps=0.8, minpts=20),
    DataSize(n_samples=2**9, n_features=2): Params(eps=0.15, minpts=4),
    DataSize(n_samples=2**9, n_features=3): Params(eps=0.1545, minpts=6),
    DataSize(n_samples=2**9, n_features=10): Params(eps=0.7, minpts=20),
    DataSize(n_samples=2**10, n_features=2): Params(eps=0.1066, minpts=4),
    DataSize(n_samples=2**10, n_features=3): Params(eps=0.26, minpts=6),
    DataSize(n_samples=2**10, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**11, n_features=2): Params(eps=0.095, minpts=4),
    DataSize(n_samples=2**11, n_features=3): Params(eps=0.18, minpts=6),
    DataSize(n_samples=2**11, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**12, n_features=2): Params(eps=0.0715, minpts=4),
    DataSize(n_samples=2**12, n_features=3): Params(eps=0.17, minpts=6),
    DataSize(n_samples=2**12, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**13, n_features=2): Params(eps=0.073, minpts=4),
    DataSize(n_samples=2**13, n_features=3): Params(eps=0.149, minpts=6),
    DataSize(n_samples=2**13, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**14, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2**14, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2**14, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**15, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2**15, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2**15, n_features=10): Params(eps=0.6, minpts=20),
    DataSize(n_samples=2**16, n_features=2): Params(eps=0.0695, minpts=4),
    DataSize(n_samples=2**16, n_features=3): Params(eps=0.108, minpts=6),
    DataSize(n_samples=2**16, n_features=10): Params(eps=0.6, minpts=20),
}

def gen_rand_data(n_samples, n_features, centers=10, dtype=np.float64):
    X, *_ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=SEED,
    )
    X = StandardScaler().fit_transform(X)

    data_size = DataSize(n_samples=n_samples, n_features=n_features)
    params = OPTIMAL_PARAMS.get(
        data_size, Params(eps=DEFAULT_EPS, minpts=DEFAULT_MINPTS)
    )

    return (X.flatten().astype(dtype), params.eps, params.minpts)


def gen_data(n_samples, n_features, centers=10, dtype=np.float64):
    X, eps, minpts = gen_rand_data(n_samples, n_features, centers, dtype)
    return X, eps, minpts

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

X, eps, minpts = gen_data(16384, 10)
X = X.reshape(-1, 1)

queue = SyclQueue("gpu")
dpt_X = dpt.asarray(X, usm_type="device", sycl_queue=queue)

### CPU execution
pred = DBSCAN(eps=eps, min_samples=minpts).fit_predict(X)

times = []
for _ in range(5):
    tic = perf_counter()
    #pred = DBSCAN(eps=eps, min_samples=minpts).fit_predict(dpt_X)
    model = DBSCAN(eps=eps, min_samples=minpts).fit(X)
    toc = perf_counter()
    times.append(toc-tic)

avg_time = sum(times)/len(times)
print("")
print(f"DBSCAN fit on CPU in {avg_time*1000} ms")

### GPU execution
with config_context(target_offload="gpu:0"):
    pred = DBSCAN(eps=eps, min_samples=minpts).fit_predict(dpt_X)

times = []
with config_context(target_offload="gpu:0"):
    for _ in range(5):
        tic = perf_counter()
        #pred = DBSCAN(eps=eps, min_samples=minpts).fit_predict(dpt_X)
        model = DBSCAN(eps=eps, min_samples=minpts).fit(dpt_X)
        toc = perf_counter()
        times.append(toc-tic)

avg_time = sum(times)/len(times)
print("")
print(f"DBSCAN fit on GPU in {avg_time*1000} ms")


