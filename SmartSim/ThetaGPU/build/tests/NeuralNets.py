import numpy as np
from math import floor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from time import perf_counter

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
            nn.LeakyReLU(0.3), # 0.3 is the slope for x<0, could be made a hyperparameter too and passed as input
            nn.Linear(self.nNeurons, self.ndOut),
        )

    # Define the method to do a forward pass
    def forward(self, x):
        return self.net(x)


## Create Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, ndIn, ndOut): # initialize the class attributes
        self.data = data
        self.ndIn = ndIn
        self.ndOut = ndOut

    def __len__(self): # return number of samples in training data
        return len(self.data)

    def __getitem__(self,idx): # return sample from dataset at given index, splitting features and targets
        feat = self.data[idx,self.ndOut:]
        target = self.data[idx,0:self.ndOut]
        return feat, target


## Define the Training and Validation/Testing Loops
# Training Loop
def train_loop(dataloader, model, lossFn, optimizer, args):
    model.train() # set model to training mode
    num_batches = len(dataloader)

    # Loop over batches, and for each batch get the input data X and target/output data y
    totalLoss = 0.0 # total loss across all batches for this epoch
    for batch, (X, y) in enumerate(dataloader):
        # Set precision
        if (args.precision == 'float'):
            X = X.float()
            y = y.float()

        # Offload data
        if (args.device == 'cuda'):
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
    avgLoss = totalLoss/num_batches
    print(f'Avg. training loss across batches: {avgLoss:>8e}')
    print("")

    return model, avgLoss


# Testing/validating Loop
def test_loop(dataloader, model, accFn, args):
    model.eval() # set model to evaluate mode
    num_batches = len(dataloader)

    accTest = 0.0
    with torch.no_grad(): # also helpful for inference mode, reduces memory when .backward() doesn't need to be called
        for batch, (X, y) in enumerate(dataloader): # loop over batches
            # Set precision
            if (args.precision == 'float'):
                X = X.float()
                y = y.float()

            # Offload data
            if (args.device == 'cuda'):
                X = X.to(args.device)
                y = y.to(args.device)

            # Perform forward pass
            pred = model(X)
            acc = accFn(pred, y)
            accTest += acc.item()

    accTest /= num_batches
    print(f"Avg. accuracy across batches: {accTest:>8e}")
    print("")
    
    return model, accTest



## Define the Training Driver function
def trainNN(features, target, args, logger_conv):
    # Create an instance of the NN model
    model = NeuralNetwork(inputDim=args.nInputs, outputDim=args.nOutputs,
                          numNeurons=args.nNeurons)
    if (args.precision == 'double'):
        model.double()
    if (args.device != 'cpu'):
        model.to(args.device)
 
    # Define loss function, accuracy function and optimizer
    lossFn = nn.functional.mse_loss
    accFn = nn.functional.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Split data into a training and testing set
    randSeed = 42 # seed of random shuffle of the training and validation data
    splitTrain = 0.75 # fraction of total data used for training
    nTrain = floor(args.nSamples*splitTrain)
    nVal = args.nSamples-nTrain
    randGen = torch.Generator().manual_seed(randSeed)

    tmp = np.concatenate((target, features), axis=1)
    training_dataset = CustomDataset(tmp,args.nInputs,args.nOutputs)
    trainDataset, testDataset = torch.utils.data.random_split(training_dataset, 
                                [nTrain, nVal], generator=randGen)

    train_dataloader = DataLoader(trainDataset, batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(testDataset, batch_size=args.batch)

    # Train the NN
    for ep in range(args.Nepochs):
        print(f"\n Epoch {ep+1}\n-------------------------------")
        # Train
        rtime = perf_counter()
        model, loss = train_loop(train_dataloader, model, lossFn, 
                                 optimizer, args)
        rtime = perf_counter() - rtime
        timeStats.t_train = timeStats.t_train + rtime
       
        # Validate
        model, acc = test_loop(test_dataloader, model, accFn, args)
        
        logger_conv.info("%d %.12e %.12e",ep+1,loss,acc)
        if (loss <= args.tolerance):
            print("Loss tolerance reached!\n")
            break

    # Write NN model to file
    torch.save(model.state_dict(), './NNmodel.pt', _use_new_zipfile_serialization=False)
    print("Saved model to disk \n")

    return model, timeStats



## Define the Model Inference and Prediction Function
def predictNN(model, features, target, args):
    # Define loss function, accuracy function and optimizer
    accFn = nn.functional.mse_loss

    # Turn data into a dataset and create dataloaders
    data = np.concatenate((target, features), axis=1) 
    dataset = CustomDataset(data,args.nInputs,args.nOutputs)
    dataloader = DataLoader(dataset, batch_size=args.batch)

    # Predict data
    acc = 0.0
    num_batches = len(dataloader)
    model.eval()
    rtime = perf_counter()
    with torch.no_grad(): # helpful for inference, reduces memory when .backward() is not called
        for batch, (X, y) in enumerate(dataloader):
            # Set precision
            if (args.precision == 'float'):
                X = X.float()
                y = y.float()

            # Offload data
            if (args.device == 'cuda'):
                X = X.to(args.device)
                y = y.to(args.device)

            # Predict
            pred = model(X)
            acc += accFn(pred, y).item()
            if (batch == 0):
                preds = pred
            else:
                preds = torch.cat((preds, pred), 0)

    rtime = perf_counter() - rtime
    timeStats.t_inf = timeStats.t_inf + rtime
    acc /= num_batches
    if (args.device == 'cuda'):
        predictions = preds.cpu().numpy()
    else:
        predictions = preds.numpy()

    return predictions, acc, timeStats


class timeStats:
    t_train = 0.0 # local total training time
    t_inf = 0.0 # local total inference time
  
