import dpctl.tensor as dpt
import numpy as np
from dpctl import SyclQueue
from mpi4py import MPI

from sklearnex.spmd.basic_statistics import BasicStatistics as BasicStatisticsSpmd


def generate_data(par, size, seed=777):
    ns, nf = par["ns"], par["nf"]

    data_blocks, weight_blocks = [], []
    rng = np.random.default_rng(seed)

    for b in range(size):
        data = rng.uniform(b, (b + 1) * (b + 1), size=(ns, nf))
        weights = rng.uniform(1, (b + 1), size=ns)
        weight_blocks.append(weights)
        data_blocks.append(data)

    data = np.concatenate(data_blocks, axis=0)
    weights = np.concatenate(weight_blocks)

    return (data, weights)


q = SyclQueue("gpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

params_spmd = {"ns": 19, "nf": 31}

data, weights = generate_data(params_spmd, size)
weighted_data = np.diag(weights) @ data

dpt_data = dpt.asarray(data, usm_type="device", sycl_queue=q)
dpt_weights = dpt.asarray(weights, usm_type="device", sycl_queue=q)

gtr_mean = np.mean(weighted_data, axis=0)
gtr_std = np.std(weighted_data, axis=0)

bss = BasicStatisticsSpmd(["mean", "standard_deviation"])
bss.fit(dpt_data, dpt_weights)

print(f"Computed mean on rank {rank}:\n", bss.mean)
print(f"Computed std on rank {rank}:\n", bss.standard_deviation)
