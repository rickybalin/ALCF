import sys
import torch
import torch.nn as nn


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


def jitModel(nSamples, model_name, device):
    # Instantiate PT model and load state dict
    nInputs = 1
    nOutputs = 1
    nNeurons = 20
    model = NeuralNetwork(inputDim=nInputs, outputDim=nOutputs, numNeurons=nNeurons)
    model.load_state_dict(torch.load(f'{model_name}.pt', map_location=torch.device('cpu')))
    model.to(device)

    # Jit the model
    dummy_input = torch.rand(nSamples,1).to(device)
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model,dummy_input)
        traced_model = torch.jit.freeze(traced_model)
    traced_model.save(f'{model_name}_jit.pt')
    

def main():
    # Convert the model
    nSamples = 1800000
    model_name = 'model_fp32'
    device = 'cuda'
    jitModel(nSamples, model_name, device)


if __name__ == '__main__':
    main()
