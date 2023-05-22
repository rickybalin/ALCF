# General imports
import sys
import numpy as np
from time import perf_counter
import argparse
from math import floor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# Horovod
import horovod.torch as hvd

## Define the Neural Network Structure
class NeuralNetwork(nn.Module):
    def __init__(self, inputDim, outputDim, numNeurons):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.net = nn.Sequential(
            nn.Linear(self.ndIn, self.nNeurons),
            nn.LeakyReLU(0.3),
            nn.Linear(self.nNeurons, self.ndOut),
        )

    def forward(self, x):
        return self.net(x)


## Create Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, ndIn, ndOut):
        self.data = data
        self.ndIn = ndIn
        self.ndOut = ndOut

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        feat = self.data[idx,self.ndOut:]
        target = self.data[idx,0:self.ndOut]
        return feat, target


## Define the Training and Validation/Testing Loops
# Training Loop
def train_loop(dataloader, model, lossFn, optimizer, args, hrank):
    model.train() # set model to training mode
    num_batches = len(dataloader)

    # Loop over batches, and for each batch get the input data X and target/output data y
    totalLoss = 0.0 # total loss across all batches for this epoch
    for batch, (X, y) in enumerate(dataloader):
        # Offload data
        if (args.device != 'cpu'):
            X = X.to(args.device)
            y = y.to(args.device)

        # Forward and backward pass
        optimizer.zero_grad()
        pred = model(X)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    # Print out average, min and max loss across batches
    totalLoss = totalLoss/num_batches
    avgLoss = metric_average(totalLoss, 'loss_avg')

    return model, avgLoss

# Validating Loop
def val_loop(dataloader, model, accFn, args, hrank):
    model.eval() # set model to evaluate mode
    num_batches = len(dataloader)

    accVal = 0.0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader): # loop over batches
            # Offload data
            if (args.device != 'cpu'):
                X = X.to(args.device)
                y = y.to(args.device)

            # Perform forward pass
            pred = model(X)
            acc = accFn(pred, y)
            accVal += acc.item()

    accVal /= num_batches
    avgVal = metric_average(accVal, 'val_avg')

    return model, avgVal


## Define the Training Driver function
def trainNN(features, target, args, hrank, hsize):
    # Create an instance of the NN model
    model = NeuralNetwork(inputDim=args.nInputs, outputDim=args.nOutputs,
                          numNeurons=args.nNeurons)
    if (args.precision == 'fp64'):
        model.double()
    if (args.device != 'cpu'):
        model.to(args.device)

    # Define loss function, accuracy function and optimizer
    lossFn = nn.functional.mse_loss
    accFn = nn.functional.mse_loss
    lr = args.learning_rate * hsize
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Split data into a training and validaiton set
    randSeed = 42 # seed of random shuffle of the training and validation data
    splitTrain = 0.75 # fraction of total data used for training
    nTrain = floor(args.nSamples*splitTrain)
    nVal = args.nSamples-nTrain
    randGen = torch.Generator().manual_seed(randSeed)

    tmp = np.concatenate((target, features), axis=1)
    if (args.precision == 'fp32'):
        tmp = torch.tensor(tmp, dtype=torch.float32)
    elif (args.precision == 'fp64'):
        tmp = torch.tensor(tmp, dtype=torch.float64)
    training_dataset = CustomDataset(tmp,args.nInputs,args.nOutputs)
    trainDataset, valDataset = torch.utils.data.random_split(training_dataset,
                                [nTrain, nVal], generator=randGen)

    # Data parallel loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                           trainDataset, num_replicas=hsize, rank=hrank)
    train_dataloader = DataLoader(trainDataset, batch_size=args.batch,
                                  sampler=None)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
                           valDataset, num_replicas=hsize, rank=hrank)
    val_dataloader = DataLoader(valDataset, batch_size=args.batch,
                                sampler=None)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         num_groups=1)

    # Train the NN
    t_train = 0.0
    t_val = 0.0
    t_tp = 0.0
    v_tp = 0.0
    for ep in range(args.nEpochs):
        if (hrank==0):
            print(f"\nEpoch {ep+1}\n-------------------------------")
            sys.stdout.flush()

        # Train
        rtime = perf_counter()
        model, loss = train_loop(train_dataloader, model, lossFn,
                                 optimizer, args, hrank)
        rtime = perf_counter() - rtime
        if (ep>0):
            t_train = t_train + rtime
            local_t_tp = nTrain/rtime
            global_t_tp = sum_across_ranks(local_t_tp,'sum')
            t_tp = t_tp + global_t_tp
            if (hrank==0):
               print(f'average loss: {loss:.6e}')
               print(f'[0]: train throughput: {local_t_tp:.6e}')
               print(f'[0]: train time: {rtime:.6e}')
               print(f'global train throughput: {global_t_tp:.6e}')

        # Validate
        rtime = perf_counter()
        model, acc = val_loop(val_dataloader, model, accFn, args, hrank)
        rtime = perf_counter() - rtime
        if (ep>0):
            t_val = t_val + rtime
            local_v_tp = nVal/rtime
            global_v_tp = sum_across_ranks(local_v_tp,'sum')
            v_tp = v_tp + global_v_tp
            if (hrank==0):
               print(f'average accuracy: {acc:.6e}')
               print(f'[0]: validation throughput: {local_v_tp:.6e}')
               print(f'[0]: validation time: {rtime:.6e}')
               print(f'global validation throughput: {global_v_tp:.6e}')

        sys.stdout.flush()
        if (loss <= args.tolerance):
            break

    return model, t_train, t_val, t_tp, v_tp


## Average across hvd ranks
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name, average=True)
    return avg_tensor.item()


## Sum across hvd ranks
def sum_across_ranks(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name, average=False)
    return avg_tensor.item()


## Main
def main():
    # Horovod import and init
    hvd.init()
    hrank = hvd.rank()
    hsize = hvd.size()
    hrankl = hvd.local_rank()

    # Print HVD Info
    print(f'Horovod: I am worker {hrank}/{hsize} and local worker {hrankl}')
    hvd.allreduce(torch.tensor(0), name='barrier')
    sys.stdout.flush()

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name',default='idk',help='Node being used')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    parser.add_argument('--batch',default=64,type=int,help='Batch size')
    parser.add_argument('--precision',default='fp32',help='Precision to be used for training and inference')
    parser.add_argument('--tolerance',default=7.0e-5,help='Tolerance on loss function')
    parser.add_argument('--nEpochs',default=20,type=int,help='Number of epochs to train for')
    parser.add_argument('--learning_rate',default=0.001,help='Learning rate')
    parser.add_argument('--nSamples',default=100000,type=int,help='Number of training and inference samples')
    parser.add_argument('--nNeurons',default=20,type=int,help='Number of neurons in network layer')
    parser.add_argument('--nInputs',default=6,type=int,help='Number of model input features')
    parser.add_argument('--nOutputs',default=6,type=int,help='Number of model output targets')
    args = parser.parse_args()

    # Get/Set inter and intra op threads
    torch.set_num_threads(1)
    #torch.set_num_interop_threads(1)
    if (hrank==0):
       print(f'\nIntra-op threads: {torch.get_num_threads()}') # intra-op
       print(f'Inter-op threads: {torch.get_num_interop_threads()}\n') #inter-op

    # Start timer for entire program
    t_start = perf_counter()

    # Set device to run on
    device = torch.device(args.device)
    if (args.device=='cuda'):
        if torch.cuda.is_available():
            if (torch.cuda.device_count()>1):
                torch.cuda.set_device(hrankl)
            else:
                torch.cuda.set_device(0)
    if (hrank==0):
        print(f'Running on device: {args.device}\n')
        sys.stdout.flush()

    # Load the test data
    inputs = np.random.rand(args.nSamples,args.nInputs)
    outputs = np.random.rand(args.nSamples,args.nOutputs)

    # Train and output model
    t_start_train = perf_counter()
    model, t_train, t_val, tp_t, tp_v = trainNN(inputs, outputs, args, hrank, hsize )
    t_end_train = perf_counter()

    # End timer for entire program
    t_end = perf_counter()

    # Print some performance information
    t_run = t_end - t_start
    t_run_ave = metric_average(t_run,'average')
    t_train_tot = t_end_train - t_start_train
    t_train_tot_ave = metric_average(t_train_tot,'average')
    t_train_ave = metric_average(t_train,'average')
    t_val_ave = metric_average(t_val,'average')

    tp_train_tot = args.nSamples*args.nEpochs*hsize/t_train_tot_ave
    tp_train_global = tp_t/(args.nEpochs-1)
    tp_val_global = tp_v/(args.nEpochs-1)
    #tp_train = args.nSamples*0.75*(args.nEpochs-1)/t_train
    #tp_train_ave = metric_average(tp_train,'average')
    #tp_val = args.nSamples*0.25*(args.nEpochs-1)/t_val
    #tp_val_ave = metric_average(tp_val,'average')
    if (hrank==0):
        print("\nPerformance info (average across ranks):")
        print(f"Total run time: {t_run_ave:.6e}")
        print(f"Total train time: {t_train_tot_ave:.6e}")
        print(f"Total train loop time: {t_train_ave:.6e}")
        print(f"Total validation loop time: {t_val_ave:.6e}")
        print(f"Global throughput based on total train time: {tp_train_tot:.6e}")
        print(f"Global train loop throughput: {tp_train_global:.6e}")
        print(f"Global validation loop throughput: {tp_val_global:.6e}")
        #print(f"Throughput based on total train loop time: {tp_train_ave}")
        #print(f"Throughput based on total val loop time: {tp_val_ave}")
        sys.stdout.flush()



## Run main
if __name__ == '__main__':
    main()


