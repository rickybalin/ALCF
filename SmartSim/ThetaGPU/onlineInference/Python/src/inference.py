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

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    args = parser.parse_args()

    # Initialize SmartRedis clients
    client = init_client(args.dbnodes)
    comm.Barrier()
    if (rank==0):
        print('All SmartRedis clients initialized')

    # Load model onto Orchestrator
    if (rank==0):
        client.set_model_from_file('model', './model_jit.pt', 'TORCH', device='GPU')
        print('Uploaded model to Orchestrator')
    comm.Barrier()

    # Set parameters for array of random numbers to be set as inference data
    # In this example we create inference data for a simple function
    # y=f(x), which has 1 input (x) and 1 output (y)
    # The domain for the function is from 0 to 10
    # The inference data is obtained from a uniform distribution over the domain
    nSamples = 64
    xmin = 0.0 
    xmax = 10.0

    # Generate the key for the inference data
    # The key will be tagged with the rank ID
    inf_key = 'x.'+str(rank)
    pred_key = 'p.'+str(rank)

    # Open file to write predictions
    if (rank==0):
        fid = open('./predictions.dat', 'w')

    # Emulate integration of PDEs with a do loop
    numts = 2
    for its in range(numts):
        sleep(10)

        # Generate the input data for the polynomial y=f(x)=x**2 + 3*x + 1
        inputs = np.random.uniform(low=xmin, high=xmax, size=(nSamples,1))

        # Perform inferece
        client.put_tensor(inf_key, inputs)
        client.run_model('model', inputs=[inf_key], outputs=[pred_key])
        predictions = client.get_tensor(pred_key)
        comm.Barrier()
        if (rank==0):
            print(f'Performed inference on all ranks for step {its+1}')

        # Write predictions to file
        if (rank==0):
            truth = inputs**2 + 3*inputs + 1
            for i in range(nSamples):
                fid.write(f'{inputs[i,0]:.6e} {predictions[i,0]:.6e} {truth[i,0]:.6e}\n')

    if (rank==0):
        fid.close()


if __name__ == '__main__':
    main()
