import sys
import os
import argparse
from time import sleep, perf_counter
import numpy as np
import logging
from math import floor
from smartredis import Client

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file,mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def init_client(args, logger_init):
    if (args.dbnodes==1):
        tic = perf_counter()
        client = Client(cluster=False)
        toc = perf_counter()
    else:
        tic = perf_counter()
        client = Client(cluster=True)
        toc = perf_counter()
    t_init = toc - tic
    if (args.logging=='verbose'):
        logger_init.info('%.8e',t_init)
    return client, t_init

def main():
    # Import and initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    #print(f'Rank {rank}/{size} says hello from node {name}') 
    #comm.Barrier()
    #sys.stdout.flush()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--ppn',default=4,type=int,help='Number of processes per node')
    parser.add_argument('--logging',default='no',help='Level of performance logging')
    args = parser.parse_args()

    # Create log files
    t_meta = 0.
    if (args.logging=='verbose'):
        logger_init = setup_logger('client_init', f'client_init_{rank}.log')
        logger_meta = setup_logger('meta', f'meta_data_{rank}.log')
        logger_data = setup_logger('train_data', f'train_data_{rank}.log')
    elif (args.logging=='verbose-perf'):
        if (rank==0):
            logger_init = setup_logger('client_init', 'client_init.log')
            logger_meta = setup_logger('meta', 'meta_data.log')
            logger_data = setup_logger('train_data', 'train_data.log')
            logger_loop = setup_logger('main-loop', 'loop.log')
        else:
            logger_init = None
    elif (args.logging=='fom'):
        if (rank==0):
            logger_fom = setup_logger('fom', 'fom.log')
        else:
            logger_init = None
    else:
        logger_init = None

    # Initialize SmartRedis clients
    client, t_init = init_client(args, logger_init)
    comm.Barrier()
    if (rank==0):
        print('All SmartRedis clients initialized')
        sys.stdout.flush()
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
    nSamples = 20000
    xmin = 0.0 
    xmax = 10.0
    nInputs = 1
    nOutputs = 1

    # Send array used to communicate whether to keep running data loader or ML
    if ((rank%args.ppn) == 0):
        arrMLrun = np.array([1, 1])
        tic = perf_counter()
        client.put_tensor('check-run', arrMLrun)
        toc = perf_counter()
        t_meta = t_meta + (toc - tic)

    # Send some information regarding the training data size
    if ((rank%args.ppn) == 0):
        arrInfo = np.array([nSamples, nInputs+nOutputs, nInputs,
                            size, args.ppn, rank])
        tic = perf_counter()
        client.put_tensor('sizeInfo', arrInfo)
        toc = perf_counter()
        t_meta = t_meta + (toc - tic)
        print(f'[{rank}]: Sent size info of training data to database')
        sys.stdout.flush()

    # Emulate integration of PDEs with a do loop
    if (rank==0):
        print('Starting computation loop ...')
        sys.stdout.flush()
    numts = 1000
    stepInfo = np.zeros(2, dtype=int)
    t_train = np.zeros([numts,])
    tic_l = perf_counter()
    for its in range(numts):
        # Need to sleep a few sec or this loop is too fast
        sleep(0.5)

        # First off check if ML is done training, if so exit from loop
        tic = perf_counter()
        arrMLrun = client.get_tensor('check-run')
        toc = perf_counter()
        t_meta = t_meta + (toc - tic)
        if (arrMLrun[0]<0.5):
            break

        # Generate the training data for the polynomial y=f(x)=x**2 + 3*x + 1
        # place output in first column and input in second column
        inputs = np.random.uniform(low=xmin, high=xmax, size=(nSamples,1))
        outputs = inputs**2 + 3*inputs + 1
        sendArr = np.concatenate((outputs, inputs), axis=1)

        # Send training data to database
        send_key = 'y.'+str(rank)+'.'+str(its+1)
        if (rank==0 and args.logging!='verbose-collect'):
            print(f'Sending training data with key {send_key} and shape {sendArr.shape}')
        tic = perf_counter()
        client.put_tensor(send_key, sendArr)
        toc = perf_counter()
        t_train[its] = toc - tic

        if (args.logging!='verbose-perf'):
            comm.Barrier()
            if (rank==0):
                print(f'All ranks finished sending training data')
                sys.stdout.flush()
        
        if (args.logging=='verbose'):
            logger_data.info('%.8e',t_train[its])

        # Send the time step number, used by ML program to determine
        # when new data is available
        if ((rank%args.ppn) == 0):
            stepInfo[0] = int(its+1)
            tic = perf_counter()
            client.put_tensor('step', stepInfo)
            toc = perf_counter()
            t_meta = t_meta + (toc - tic)

    toc_l = perf_counter()
    t_loop = toc_l - tic_l

    if (args.logging=='verbose'):
        logger_meta.info('%.8e',t_meta)

    # Collect performance statistics
    if (args.logging=='verbose-perf'):
        if (rank==0):
            print('Collecting performance stats ... ')
            sys.stdout.flush()
        t_init_gather = None
        t_meta_gather = None
        t_train_gather = None
        t_loop_gather = None
        if (rank==0):
            t_init_gather = np.empty([size])
            t_meta_gather = np.empty([size])
            t_train_gather = np.empty([size,numts])
            t_loop_gather = np.empty([size])
        comm.Gather(np.array(t_init),t_init_gather,root=0)
        comm.Gather(np.array(t_meta),t_meta_gather,root=0)
        comm.Gather(t_train,t_train_gather,root=0)
        comm.Gather(np.array(t_loop),t_loop_gather,root=0)
        its = floor(its/10)*10
        if (rank==0):
            for ir in range(size):
                logger_init.info('%.8e',t_init_gather[ir])
                logger_meta.info('%.8e',t_meta_gather[ir])
                logger_loop.info('%.8e',t_loop_gather[ir])
                for i in range(its):
                    logger_data.info('%.8e',t_train_gather[ir,i])

    # Exit
    if (rank==0):
        print('Exiting ...')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
