import numpy as np
import torch
from smartredis import Client

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def f(x):
    return x**2 + 3*x + 1


# Connect a SmartRedis client to the Redis database
client = Client(cluster=False) # must be changed to True if database on >1 nodes

if rank == 0:

    # upload model to db
    fname = "model_jit.pt"
    model_key = "model"
    client.set_model_from_file(model_key, fname, "TORCH")
    
    # create input data
    inputs = np.random.uniform(low=0, high=10, size=(64,1))
    input_key = "input"
    client.put_tensor(input_key, inputs)
    
    # perform inference on db and return outputs
    output_key = "output"
    client.run_model(model_key, input_key, output_key)
    results = client.get_tensor(output_key)

    inputs = inputs.flatten()
    results = results.flatten()

    # print results
    for x,y in zip(inputs, results):
        print(x,y)
    
    # plot results
    import matplotlib.pyplot as plt
    plt.plot(inputs, results, '.')
    x = np.linspace(0, 10, 100)
    plt.plot(x, f(x), '-r')
    plt.savefig('fig.pdf', bbox_inches='tight')
    
