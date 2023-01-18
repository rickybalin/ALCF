import sys
import argparse
from time import sleep, perf_counter
import numpy as np
import logging
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file,mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def init_onnx_session(args, model, logger_init):
    ort_providers = ['CPUExecutionProvider'] # always include CPU execution provider
    if str(args.model_device)=='cuda':
        # NB: ORDER MATTERS!!! CUDA before CPU EP if want CUDA to be dafault
        ort_providers = ['CUDAExecutionProvider'] + ort_providers
    tic = perf_counter()
    options = ort.SessionOptions()
    ort_session = ort.InferenceSession(model, sess_options=options,
                                       providers=ort_providers)
    toc = perf_counter()
    t_init = toc - tic
    if (args.logging=='verbose'):
        logger_init.info('%.8e',t_init)
    return ort_session, t_init


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


def convert_pt2onnx(nSamples, model_name, model_check=False):
    # Instantiate PT model and load state dict
    nInputs = 1
    nOutputs = 1
    nNeurons = 20
    model = NeuralNetwork(inputDim=nInputs, outputDim=nOutputs, numNeurons=nNeurons)
    model.load_state_dict(torch.load(f'{model_name}.pt', map_location=torch.device('cpu')))
    model.eval()

    # Export the model to ONNX
    inf_batch = nSamples
    dummy_input = torch.randn(nSamples, nInputs, dtype=torch.float32)
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, dummy_input, f'{model_name}.onnx', verbose=False,
                      export_params=True, input_names=input_names, output_names=output_names)

    # Verify the conversion
    if (model_check):
        model_onnx = onnx.load('model.onnx')
        onnx.checker.check_model(model_onnx)
        print(onnx.helper.printable_graph(model_onnx.graph))
        sys.stdout.flush()


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
    parser.add_argument('--model_device',default='cpu',help='Device to run model on')
    parser.add_argument('--sim_device',default='cpu',help='Device to run "simulation" on')
    parser.add_argument('--ppn',default=2,type=int,help='Number of processes per node')
    parser.add_argument('--logging',default='no',help='Level of performance logging')
    args = parser.parse_args()

    # Create log files
    if (args.logging=='verbose'):
        logger_init = setup_logger('session_init', f'session_init_{rank}.log')
        logger_inf = setup_logger('inference', f'inference_{rank}.log')
    elif (args.logging=='verbose-perf'):
        if (rank==0):
            logger_init = setup_logger('session_init', 'session_init.log')
            logger_inf = setup_logger('inference', 'inference.log')
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

    # Initialize ONNX inference session
    model_name = 'model'
    nSamples = 2
    convert_pt2onnx(nSamples, model_name)
    ort_session, t_init = init_onnx_session(args, f'{model_name}.onnx', logger_init)
    comm.Barrier()
    if (rank==0):
        print('Initialixed ONNX inference session')
        sys.stdout.flush()

    # Set parameters for array of random numbers to be set as inference data
    # In this example we create inference data for a simple function
    # y=f(x), which has 1 input (x) and 1 output (y)
    # The domain for the function is from 0 to 10
    # The inference data is obtained from a uniform distribution over the domain
    local_rank = rank % args.ppn
    rank_fact = rank/args.ppn
    if (rank_fact<1):
        xmin = 0.0 
        xmax = 5.0
    elif (rank_fact>=1):
        xmin = 5.0
        xmax = 10.0

    # Open file to write predictions
    if (args.logging!='verbose-perf' and args.logging!='fom'):
        if (rank%args.ppn==0):
            fname = f'./predictions_node{rank//args.ppn+1}.dat'
            fid = open(fname, 'w')

    # Emulate integration of PDEs with a do loop
    numts = 10
    t_inf = np.empty([numts,])
    tic_l = perf_counter()
    for its in range(numts):
        # Generate the input data for the polynomial y=f(x)=x**2 + 3*x + 1
        inputs = np.float32(np.random.uniform(low=xmin, high=xmax, size=(nSamples,1)))
        if (args.sim_device=='cuda'):
            # Setup
            if (its==0):
                ort_in_value = ort.OrtValue.ortvalue_from_numpy(inputs, device_type=args.sim_device, 
                                                       device_id=local_rank)
                ort_out_value = ort.OrtValue.ortvalue_from_shape_and_type([nSamples,1], np.float32,
                                                       device_type=args.sim_device, device_id=local_rank)
                io_bind = ort_session.io_binding()
                io_bind.bind_input(name='input', device_type=ort_in_value.device_name(), device_id=local_rank, 
                                element_type=np.float32, shape=ort_in_value.shape(), 
                                buffer_ptr=ort_in_value.data_ptr())
                io_bind.bind_output(name='output', device_type=ort_out_value.device_name(), device_id=local_rank, 
                                element_type=np.float32, shape=ort_out_value.shape(), 
                                buffer_ptr=ort_out_value.data_ptr())
            # Update input array on device
            ort_in_value.update_inplace(inputs)

        # Perform inferece
        if (args.sim_device=='cpu' and args.model_device=='cpu'):
            tic_i = perf_counter()
            predictions = ort_session.run([], {'input':inputs})[0]
            toc_i = perf_counter()
        elif (args.sim_device=='cpu' and args.model_device=='cuda'):
            if (its==0):
                io_bind = ort_session.io_binding()
            tic_i = perf_counter()
            io_bind.bind_cpu_input('input',inputs)
            io_bind.bind_output('output')
            ort_session.run_with_iobinding(io_bind)
            predictions = io_bind.copy_outputs_to_cpu()[0]
            toc_i = perf_counter()
        elif (args.sim_device=='cuda' and args.model_device=='cuda'):
            tic_i = perf_counter()
            ort_session.run_with_iobinding(io_bind)
            toc_i = perf_counter()
            predictions = io_bind.copy_outputs_to_cpu()[0]

        t_inf[its] = toc_i - tic_i
        
        # Print info to stdout
        if (args.logging!='verbose-perf' and args.logging!='fom'):
            comm.Barrier()
            if (rank==0):
                print(f'Performed inference on all ranks for step {its+1}')
                sys.stdout.flush()
        
        # Print timings with old (not performant) approach
        if (args.logging=='verbose'):
            logger_inf.info('%.8e',toc_i-tic_i)

        # Write predictions to file
        if (args.logging!='fom' and args.logging!='verbose-perf'):
            if (rank%args.ppn==0):
                truth = inputs**2 + 3*inputs + 1
                for i in range(nSamples):
                    fid.write(f'{inputs[i,0]:.6e} {predictions[i,0]:.6e} {truth[i,0]:.6e}\n')
    
    toc_l = perf_counter()
    t_loop = toc_l - tic_l

    # Compute FOM
    fom = (t_loop) / numts
    if (args.logging=='fom'):
        fom_avg = comm.allreduce(fom, op=MPI.SUM)
        fom_avg = fom_avg / size
        if (rank==0):
            logger_fom.info('%.8e',fom_avg)

    # Collect performance statistics
    if (args.logging=='verbose-perf'):
        if (rank==0):
            print('Collecting performance stats ... ')
            sys.stdout.flush()
        t_init_gather = None
        t_inf_gather = None
        t_loop_gather = None
        if (rank==0):
            t_init_gather = np.empty([size])
            t_inf_gather = np.empty([size,numts])
            t_loop_gather = np.empty([size])
        comm.Gather(np.array(t_init),t_init_gather,root=0)
        comm.Gather(t_inf,t_inf_gather,root=0)
        comm.Gather(np.array(t_loop),t_loop_gather,root=0)
        if (rank==0):
            for ir in range(size):
                logger_init.info('%.8e',t_init_gather[ir])
                logger_loop.info('%.8e',t_loop_gather[ir])
                for its in range(numts):
                    logger_inf.info('%.8e',t_inf_gather[ir,its])
    
    if (rank%args.ppn==0):
        fid.close()

    # Exit
    if (rank==0):
        print('Exiting ...')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
