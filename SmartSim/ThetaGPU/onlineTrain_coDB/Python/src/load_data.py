import os
import argparse
from time import sleep
import numpy as np
from smartredis import Client

def init_client(nnDB):
    if (nnDB==1):
        client = Client(cluster=False)
    else:
        client = Client(cluster=True)
    return client

def main():
    # Import and initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    print(f'Rank {rank}/{size} says hello from node {name}') 
    comm.Barrier()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--ppn',default=4,type=int,help='Number of processes per node')
    args = parser.parse_args()
    if (rank==0):
        print(f'\nRunning with {args.dbnodes} DB nodes')
        print(f'and with {args.ppn} processes per node \n')

    # Initialize SmartRedis clients
    client = init_client(args.dbnodes)
    comm.Barrier()
    if (rank==0):
        print('All SmartRedis clients initialized')
    if ((rank%args.ppn) == 0):
        SSDB = os.getenv('SSDB')
        f = open(f'SSDB_{name}.dat', 'w')
        f.write(f'{SSDB}')
        f.close()
    
    # Set parameters for array of random numbers to be set as inference data
    # In this example we create inference data for a simple function
    # y=f(x), which has 1 input (x) and 1 output (y)
    # The domain for the function is from 0 to 10
    # The inference data is obtained from a uniform distribution over the domain
    nSamples = 64
    xmin = 0.0 
    xmax = 10.0
    nInputs = 1
    nOutputs = 1

    # Send array used to communicate whether to keep running data loader or ML
    if ((rank%args.ppn) == 0):
        arrMLrun = np.array([1, 1])
        client.put_tensor('check-run', arrMLrun)

    # Send some information regarding the training data size
    if ((rank%args.ppn) == 0):
        arrInfo = np.array([nSamples, nInputs+nOutputs, nInputs,
                            size, args.ppn, rank])
        client.put_tensor('sizeInfo', arrInfo)
        print('Sent size info of training data to database')

    # Emulate integration of PDEs with a do loop
    numts = 1000
    stepInfo = np.zeros(2, dtype=int)
    for its in range(numts):
        # Sleep for a few seconds to emulate the time required by PDE integration
        sleep(2)

        # First off check if ML is done training, if so exit from loop
        arrMLrun = client.get_tensor('check-run')
        if (arrMLrun[0]<0.5):
            break

        # Generate the training data for the polynomial y=f(x)=x**2 + 3*x + 1
        # place output in first column and input in second column
        inputs = np.random.uniform(low=xmin, high=xmax, size=(nSamples,1))
        outputs = inputs**2 + 3*inputs + 1
        sendArr = np.concatenate((outputs, inputs), axis=1)

        # Send training data to database
        send_key = 'y.'+str(rank)+'.'+str(its+1)
        if (rank==0):
            print(f'Sending training data with key {send_key} and shape {sendArr.shape}')
        client.put_tensor(send_key, sendArr)
        comm.Barrier()
        if (rank==0):
            print(f'All ranks finished sending training data')

        # Send the time step number, used by ML program to determine
        # when new data is available
        if ((rank%args.ppn) == 0):
            stepInfo[0] = int(its+1)
            client.put_tensor('step', stepInfo)

    if (rank==0):
        print('Exiting ...')


if __name__ == '__main__':
    main()
