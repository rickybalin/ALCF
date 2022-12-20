import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import sleep, perf_counter
from os.path import exists
import horovod.torch as hvd

from smartredis import Client

## Define logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file,mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

## Initialize Redis clients
def init_client(SSDB, args, logger_init):
    if (args.dbnodes==1):
        tic = perf_counter()
        client = Client(address=SSDB,cluster=False)
        toc = perf_counter()
    else:
        tic = perf_counter()
        client = Client(address=SSDB,cluster=True)
        toc = perf_counter()
    t_init = toc - tic
    if (args.logging=='verbose'):
        logger_init.info('%.8e',toc-tic)
    return client, t_init

## Define the Neural Network Structure
class NeuralNetwork(nn.Module):
    # The class takes as inputs the input and output dimensions and the number of layers   
    def __init__(self, inputDim, outputDim, numNeurons):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.net = nn.Sequential(
            nn.Linear(self.ndIn, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.nNeurons),
            nn.ReLU(),
            nn.Linear(self.nNeurons, self.ndOut),
        )

    # Define the method to do a forward pass
    def forward(self, x):
        return self.net(x)


## Define Datasets
class RankDataset(torch.utils.data.Dataset):
    # contains the keys of all tensors uploaded to db by phasta ranks
    def __init__(self, num_tot_tensors, step_num, head_rank):
        self.total_data = num_tot_tensors
        self.step = step_num
        self.head_rank = head_rank

    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        tensor_num = idx+self.head_rank
        return f"y.{tensor_num}.{self.step}"

class MinibDataset(torch.utils.data.Dataset):
    #dataset of each ML rank in one epoch with the concatenated tensors
    def __init__(self,concat_tensor):
        self.concat_tensor = concat_tensor

    def __len__(self):
        return len(self.concat_tensor)

    def __getitem__(self, idx):
        return self.concat_tensor[idx]


## Training subroutine
def train(model, train_sampler, train_tensor_loader, optimizer, epoch, 
          batch, ndOut, client, args, logger_data):
    model.train()
    running_loss = 0.0
    n_tensors = len(train_tensor_loader)
    t_train = np.empty([n_tensors,])
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    loss_fn = nn.functional.mse_loss

    for tensor_idx, tensor_keys in enumerate(train_tensor_loader):
        # grab data from database
        if (args.logging != 'verbose-perf'):
            print(f'[{hvd.rank()}]: Grabbing tensors with key {tensor_keys}')
            hvd.allreduce(torch.tensor(0), name='barrier')
            sys.stdout.flush()
        tic = perf_counter()
        concat_tensor = torch.cat([torch.from_numpy(client.get_tensor(key)) \
                      for key in tensor_keys], dim=0)
        toc = perf_counter()
        t_train[tensor_idx] = toc - tic
        if (args.logging=='verbose'):
            logger_data.info('%.8e',toc-tic)
        concat_tensor = concat_tensor.float()

        mbdata = MinibDataset(concat_tensor)
        train_loader = torch.utils.data.DataLoader(mbdata, shuffle=True, batch_size=batch)
        for batch_idx, dbdata in enumerate(train_loader):
            # split inputs and outputs
            if (args.device != 'cpu'):
               dbdata = dbdata.to(args.device)
            target = dbdata[:, :ndOut]
            features = dbdata[:, ndOut:]

            optimizer.zero_grad()
            output = model.forward(features)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    running_loss = running_loss / len(train_loader) / batch
    loss_avg = metric_average(running_loss, 'running_loss')

    if hvd.rank() == 0:
        print(f"Training set: Average loss: {loss_avg:>8e}")

    return model, loss_avg, t_train


## Average across hvd ranks
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


## Main
def main():
    # Horovod import and initialization
    #import horovod.torch as hvd
    from horovod.torch.mpi_ops import Sum
    hvd.init()
    hrank = hvd.rank()
    hrankl = hvd.local_rank()
    hsize = hvd.size()
    #print(f'HVD rank {hrank}/{hsize} and local rank {hrankl}')
    #sys.stdout.flush()

    # MPI import and initialization
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    #print(f'Rank {rank}/{size} says hello from node {name}')
    #sys.stdout.flush()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dbnodes',default=1,type=int,help='Number of database nodes')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    parser.add_argument('--ppn',default=2,type=int,help='Number of processes per node')
    parser.add_argument('--logging',default='no',help='Level of performance logging')
    args = parser.parse_args()

    # Create log files
    if (args.logging=='verbose'):
        logger_init = setup_logger('client_init', f'client_init_ml_{rank}.log')
        logger_meta = setup_logger('meta', f'meta_data_ml_{rank}.log')
        logger_data = setup_logger('train_data', f'train_data_ml_{rank}.log')
    elif (args.logging=='verbose-perf'):
        if (rank==0):
            logger_init = setup_logger('client_init', 'client_init_ml.log')
            logger_meta = setup_logger('meta', 'meta_data_ml.log')
            logger_data = setup_logger('train_data', 'train_data_ml.log')
            logger_loop = setup_logger('main-loop', 'loop_ml.log')
        else:
            logger_init = None
            logger_data = None
    else:
        logger_init = None
        logger_data = None

    # Read the address of the co-located database first
    SSDB_file = f'SSDB_{name}.dat'
    c = 0
    while True:
        if (exists(SSDB_file)):
            f = open(SSDB_file, "r")
            SSDB = f.read()
            f.close()
            if (SSDB == ''):
                continue
            else: 
                #print(f'[{rank}]: read SSDB={SSDB}')
                sys.stdout.flush()
                break
        else:
            if (c==0):
                #print(f'[{rank}]: WARNING, looked for {SSDB_file} but did not find it')
                sys.stdout.flush()
            c+=1
            continue
    comm.Barrier()

    # Initialize Redis clients on each rank
    #print(f'[{rank}]: connecting to SSDB={SSDB}')
    client, t_init = init_client(SSDB, args, logger_init)
    comm.Barrier()
    if (hrank == 0):
        print("All Python clients initialized\n")
        sys.stdout.flush()

    # Pull metadata from database
    t_meta = 0.
    while True:
        if (client.poll_tensor("sizeInfo",0,1)):
            tic = perf_counter()
            dataSizeInfo = client.get_tensor('sizeInfo')
            toc = perf_counter()
            t_meta = t_meta + (toc-tic)
            break
    comm.Barrier()
    if (hrank == 0):
        print("Retreived metadata from DB \n")
        sys.stdout.flush()

    npts = dataSizeInfo[0]
    ndTot = dataSizeInfo[1]
    ndIn = dataSizeInfo[2]
    ndOut = ndTot - ndIn
    num_tot_tensors = dataSizeInfo[3]
    num_db_tensors = dataSizeInfo[4]
    head_rank = dataSizeInfo[5]
    if (args.logging!='verbose-perf'):
        print(f'[{rank}]: head rank is {head_rank}')
        sys.stdout.flush()

    # NN Training Hyper-Parameters
    Nepochs = 50 # number of epochs
    batch =  int(num_db_tensors/args.ppn) # how many tensors to grab from db
    mini_batch = 512 # batch size once tensors obtained from db and concatenated 
    learning_rate = 0.001*hsize # learning rate
    nNeurons = 20 # number of neuronsining settings
    tol = 1.0e-7 # convergence tolerance on loss function

    # Set device to run on
    if (hrank == 0):
        print(f"Running on device: {args.device} \n")
    device = torch.device(args.device)
    if (args.device == 'cuda'):
        if torch.cuda.is_available():
            torch.cuda.set_device(hrankl)

    # Instantiate the NN model and optimizer
    model = NeuralNetwork(inputDim=ndIn, outputDim=ndOut, numNeurons=nNeurons)
    if (args.device != 'cpu'):
        model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training setup and variable initialization
    istep = 0 # initialize the simulation step number to 0
    iepoch = 1 # epoch number
    torch.set_num_threads(1)

    # While loop that checks when training data is available on database
    if (hrank == 0):
        print("Starting training loop ... \n")
        sys.stdout.flush()
    
    tic_l = perf_counter()
    while True:
        # check to see if the time step number has been sent to database, if not cycle
        if (client.poll_tensor("step",0,1)):
            tic = perf_counter()
            tmp = client.get_tensor('step')
            toc = perf_counter()
            t_meta = t_meta + (toc-tic)
        else:
            continue

        # new data is available in database so update it and train 1 epoch
        if (istep != tmp[0]): 
            istep = tmp[0]
            if (hrank == 0):
                print("\nGetting new training data from DB ...")
                print(f"Working on time step {istep} \n")
                sys.stdout.flush()

            datasetTrain = RankDataset(num_db_tensors,istep,head_rank)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                        datasetTrain, num_replicas=args.ppn, rank=hrankl, drop_last=False)
            train_tensor_loader = torch.utils.data.DataLoader(
                        datasetTrain, batch_size=batch, sampler=train_sampler)
       
            if (iepoch==1):
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)
                optimizer = hvd.DistributedOptimizer(
                                 optimizer,named_parameters=model.named_parameters(),op=Sum)
        
            if (hrank == 0):
                print(f"\n Epoch {iepoch}\n-------------------------------")
                sys.stdout.flush()
        
            model, global_loss, tmp = train(model, train_sampler, train_tensor_loader, optimizer,
                                       iepoch, mini_batch, ndOut, client, args, logger_data)
            sys.stdout.flush()
            if (iepoch==1):
                t_train = tmp
            else:
                t_train = np.concatenate((t_train, tmp))          

            # check if tolerance on loss is satisfied
            if (global_loss <= tol):
                if (hrank == 0):
                    print("Convergence tolerance met. Stopping training loop. \n")
                    sys.stdout.flush()
                break
        
            # check if max number of epochs is reached
            if (iepoch >= Nepochs):
                if (hrank == 0):
                    print("Max number of epochs reached. Stopping training loop. \n")
                    sys.stdout.flush()
                break

            iepoch = iepoch + 1        
        
        # new data is not avalable, so train another epoch on current data
        else:
            if (hrank == 0):
                print(f"\n Epoch {iepoch}\n-------------------------------")
        
            model, global_loss, tmp = train(model, train_sampler, train_tensor_loader, optimizer,
                                       iepoch, mini_batch, ndOut, client, args, logger_data)
            t_train = np.concatenate((t_train, tmp))          
            
            # check if tolerance on loss is satisfied
            if (global_loss <= tol):
                if (hrank == 0):
                    print("Convergence tolerance met. Stopping training loop. \n")
                break
        
            # check if max number of epochs is reached
            if (iepoch >= Nepochs):
                if (hrank == 0):
                    print("Max number of epochs reached. Stopping training loop. \n")
                break

            iepoch = iepoch + 1        
            sys.stdout.flush()

    toc_l = perf_counter()
    t_loop = toc_l - tic_l

    if (args.logging=='verbose'):
        logger_meta.info('%.8e',t_meta)    

    # Save model to file before exiting
    if (hrank == 0):
        model_name = "model"
        torch.save(model.state_dict(), f"{model_name}.pt", _use_new_zipfile_serialization=False)
        # save jit traced model to be used for online inference with SmartSim
        features = np.float32(np.random.uniform(low=0, high=10, size=(100,1)))
        features = torch.from_numpy(features).to(args.device)
        module = torch.jit.trace(model, features)
        torch.jit.save(module, f"{model_name}_jit.pt")
        print("Saved model to disk\n")
        sys.stdout.flush()


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
            t_train_gather = np.empty([size,t_train.size])
            t_loop_gather = np.empty([size])
        comm.Gather(np.array(t_init),t_init_gather,root=0)
        comm.Gather(np.array(t_meta),t_meta_gather,root=0)
        comm.Gather(t_train,t_train_gather,root=0)
        comm.Gather(np.array(t_loop),t_loop_gather,root=0)
        if (rank==0):
            for ir in range(size):
                logger_init.info('%.8e',t_init_gather[ir])
                logger_meta.info('%.8e',t_meta_gather[ir])
                logger_loop.info('%.8e',t_loop_gather[ir])
                for its in range(t_train.size):
                    logger_data.info('%.8e',t_train_gather[ir,its])


    # Tell data loader to exit too
    comm.Barrier()
    if (hrank%args.ppn == 0):
        print("Telling data loader to quit ... \n")
        sys.stdout.flush()
        arrMLrun = np.zeros(2)
        client.put_tensor("check-run",arrMLrun)

    # Exit
    if (hrank == 0):
        print("Exiting ...")
        sys.stdout.flush()
    

###
if __name__ == '__main__':
    main()

