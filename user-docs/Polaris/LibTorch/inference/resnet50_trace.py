import torch
import torchvision
from time import perf_counter

device = 'cuda'

model = torchvision.models.resnet50()
model.to(device)
model.eval()

dummy_input = torch.rand(1, 3, 224, 224).to(device)

model_jit = torch.jit.trace(model, dummy_input)
tic = perf_counter()
predictions = model_jit(dummy_input)
toc = perf_counter()
print(f"Inference time: {toc-tic}")

torch.jit.save(model_jit, f"resnet50_jit.pt")
