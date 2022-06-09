# General imports
import numpy as np
from time import perf_counter
from datetime import datetime
import argparse
import logging
import torch
import torch.distributed as dist

# Import help functions
from NeuralNets_hvd import trainNN, predictNN, timeStats


## Set up logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file,mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


## Main
def main():
    # Horovod import and initialization
    import horovod.torch as hvd 
    hvd.init()
    hrank = hvd.rank()
    hsize = hvd.size()
    print(f"Horovod: I am worker {hrank} of {hsize}")
    if (hrank==0):
        print("")
    
    # Create log files
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
    logger_info = setup_logger('info', date_string+'info.log')
    logger_conv = setup_logger('convergence', date_string+'convergence.log')
    logger_time = setup_logger('time_stats', date_string+'time.log')

    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device',default='cpu',help='Device to run on')
    parser.add_argument('--batch',default=64,type=int,help='Batch size')
    parser.add_argument('--precision',default='float',help='Precision to be used for training and inference')
    parser.add_argument('--tolerance',default=7.0e-5,help='Tolerance on loss function')
    parser.add_argument('--Nepochs',default=10,type=int,help='Number of epochs to train for')
    parser.add_argument('--learning_rate',default=0.001,help='Learning rate')
    parser.add_argument('--nNeurons',default=20,type=int,help='Number of neurons in network layer')
    parser.add_argument('--nSamples',default=100000,type=int,help='Number of training and inference samples')
    parser.add_argument('--nInputs',default=6,type=int,help='Number of model input features')
    parser.add_argument('--nOutputs',default=6,type=int,help='Number of model output targets')
    args = parser.parse_args()
    if (hrank==0):
        logger_info.info("Training parameters:")
        logger_info.info("Precision: %s",args.precision)
        logger_info.info("Tolerance: %.12e",args.tolerance)
        logger_info.info("Number of epochs: %d",args.Nepochs)
        logger_info.info("Training mini-batch size: %d",args.batch)
        logger_info.info("Inference mini-batch size: %d",args.batch)
        logger_info.info("Learning rate: %.12e",args.learning_rate)
        logger_info.info("Number of neurons: %d",args.nNeurons)
        logger_info.info("Number of samples: %d",args.nSamples)
        logger_info.info("Number of inputs: %d",args.nInputs)
        logger_info.info("Number of outputs: %d",args.nOutputs)
        logger_info.info("")

    # Pin HVD process to single GPU
    if (args.device == 'cuda'):
        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())

    # Start timer for entire program
    t_start = perf_counter()

    # Set device to run on
    device = torch.device(args.device)
    if (hrank==0):
        logger_info.info('Running on device: %s\n', args.device)

    # Load the test data
    if (hrank==0):
        logger_info.info("Computing inputs and outputs ...")
    inputs = np.random.rand(args.nSamples,args.nInputs)
    outputs = np.random.rand(args.nSamples,args.nOutputs)
    if (hrank==0):
        logger_info.info("Done\n")
        print('Generated training data \n')

    # Train and output model
    if (hrank==0):
        logger_info.info("Training model ...")
        print("Training model ... \n")
    #t_start_train = perf_counter()
    model, timeStats = trainNN(inputs, outputs, args, logger_conv, hrank, hsize)
    #t_end_train = perf_counter()
    if (hrank==0):
        logger_info.info("Done\n")
        print('Done training \n')

    # Make some predictions on rank 0 only
    if (hrank==0):
        logger_info.info("Making Predictions ...")
        print("Making predictions ... \n")
        inputs = np.random.rand(args.nSamples,args.nInputs)
        outputs = np.random.rand(args.nSamples,args.nOutputs)
        #t_start_pred = perf_counter()
        predictions, accuracy, timeStats = predictNN(model, inputs, outputs, args)
        #t_end_pred = perf_counter()
        logger_info.info("Done\n")
        print('Done\n')

    # End timer for entire program
    t_end = perf_counter()

    # Print some timing information
    logger_time.info("[%d]: Total run time: %.12e", hrank, t_end - t_start)
    hvd.allreduce(torch.tensor(0), name='barrier')
    logger_time.info("[%d]: Total train time: %.12e", hrank, timeStats.t_train)
    hvd.allreduce(torch.tensor(0), name='barrier')
    logger_time.info("[%d]: Total prediction time: %.12e", hrank, timeStats.t_inf)


if __name__ == '__main__':
    main()



