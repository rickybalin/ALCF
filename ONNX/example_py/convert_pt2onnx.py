import numpy as np
from time import sleep
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

## Define the Neural Network Structure
class NeuralNetwork(nn.Module):
    def __init__(self, inputDim, outputDim, numNeurons, numLayers):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.nLayers = numLayers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.ndIn, self.nNeurons)) # input layer
        self.layers.append(nn.ReLU())
        for l in range(self.nLayers): # hidden layers
            self.layers.append(nn.Linear(self.nNeurons, self.nNeurons))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.nNeurons, self.ndOut)) # output layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


## Initialize model weights
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


## Main
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Detected device: {device}\n')
    
    # Instantiate model
    nInputs = 1
    nOutputs = 1
    nNeurons = 200
    nLayers = 25
    model = NeuralNetwork(inputDim=nInputs, outputDim=nOutputs,
                          numNeurons=nNeurons, numLayers=nLayers)
    model.apply(weights_init_uniform_rule)

    # Save model
    torch.save(model.state_dict(), 'model.pt', _use_new_zipfile_serialization=False)
    #model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
    model.to(device)
    model.eval()

    # Provide input and output names
    inf_batch = 512*100
    dummy_input = torch.randn(inf_batch,nInputs, device=device)
    print(f'Torch inputs on device: {dummy_input.device}')
    input_names = ['input']
    output_names = ['output']

    # Perform inference with torch model on device
    predictions_torch = model(dummy_input)
    print(f'Torch predictions on device: {predictions_torch.device}\n')
    #sleep(5)

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, 'model.onnx', verbose=False,
                      export_params=True, input_names=input_names, output_names=output_names)

    # Verify the conversion
    model_onnx = onnx.load('model.onnx')
    onnx.checker.check_model(model_onnx)
    #print(onnx.helper.printable_graph(model_onnx.graph))

    # Create an ONNX inference session
    #ort_device = ort.get_device()
    ort_providers = ['CPUExecutionProvider'] # always include CPU execution provider
    if str(device)=='cuda':
        ort_providers = ort_providers + ['CUDAExecutionProvider']
    options = ort.SessionOptions()
    ort_session = ort.InferenceSession('model.onnx', sess_options=options,
                                       providers=ort_providers)

    # Convert from torch tensor to numpy array
    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else: 
            return tensor.cpu().numpy()

    # Perform inference with ONNX model on CPU
    print('ONNX runtime inference on CPU')
    ort_input_name = ort_session.get_inputs()[0].name
    ort_input_value = ort.OrtValue.ortvalue_from_numpy(to_numpy(dummy_input))
    print(f'Inputs on device: {ort_input_value.device_name()}')
    #print(ort_input_value.shape())
    #print(ort_input_value.data_type())  
    ort_input = {ort_input_name: ort_input_value} # can also pass the numpy array as the value
    ort_predictions = ort_session.run(None, ort_input)
    ort_predictions_value = ort_predictions[0]
    np.testing.assert_allclose(to_numpy(predictions_torch), ort_predictions_value, rtol=1e-03, atol=1e-05)
    print('Exported model has been tested on CPU with ONNXRuntime, predictions validated to set tolerance \n')
    #sleep(5)

    # Perform inference with ONNX model on GPU
    if str(device)=='cuda':
        # Method 1: Offload input data from CPU to GPU, output data is on CPU
        print('ONNX runtime inference on GPU')
        print('Method 1: Offload input data from CPU to GPU, output data is on CPU')
        io_bind = ort_session.io_binding()
        io_bind.bind_cpu_input(ort_input_name,to_numpy(dummy_input))
        ort_output_name = ort_session.get_outputs()[0].name
        io_bind.bind_output(ort_output_name)
        # ORT will copy data over to device if 'input' is consumed by nodes with device
        ort_session.run_with_iobinding(io_bind)
        ort_predictions_gpu_value = io_bind.copy_outputs_to_cpu()[0]
        np.testing.assert_allclose(to_numpy(predictions_torch), ort_predictions_gpu_value, 
                                   rtol=1e-03, atol=1e-05)
        print('Exported model has been tested on GPU with ONNXRuntime, ',
              'predictions validated to set tolerance \n')
        sleep(5)

        # Method 2: Input data is on GPU, output data is on CPU
        print('ONNX runtime inference on GPU')
        print('Method 2: Input data is on GPU, output data is on CPU')
        ort_x_value = ort.OrtValue.ortvalue_from_numpy(to_numpy(dummy_input), device_type=str(device), 
                                                       device_id=0)
        io_bind_2 = ort_session.io_binding()
        io_bind_2.bind_input(name=ort_input_name, device_type=ort_x_value.device_name(), device_id=0, 
                                element_type=np.float32, shape=ort_x_value.shape(), 
                                buffer_ptr=ort_x_value.data_ptr())
        io_bind_2.bind_output(ort_output_name)
        ort_session.run_with_iobinding(io_bind_2)
        ort_predictions_gpu_value_2 = io_bind_2.copy_outputs_to_cpu()[0]
        np.testing.assert_allclose(to_numpy(predictions_torch), ort_predictions_gpu_value_2, 
                                   rtol=1e-03, atol=1e-05)
        print('Exported model has been tested on GPU with ONNXRuntime, ',
              'predictions validated to set tolerance \n')
        sleep(5)

        # Method 3: Input data is on GPU, output data is on GPU
        print('ONNX runtime inference on GPU')
        print('Method 3: Input data is on GPU, output data is on GPU')
        # example 3 from https://onnxruntime.ai/docs/api/python/api_summary.html#data-on-device
        #print('Exported model has been tested on GPU with ONNXRuntime, ',
        #     'predictions validated to set tolerance \n')
        


##
if __name__ == '__main__':
    main()
