import torch
from torch import nn

# input tensor dimension 64*1000
input_tensor = torch.randn(64, 1000)

# linear layer with 100 neurons
linear_layer = nn.Linear(1000, 100)

# output of the linear layer
output = linear_layer(input_tensor)

print(output.size())

# USING nn.Sequential

model = nn.Sequential(
    nn.Linear(4, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

print(model)