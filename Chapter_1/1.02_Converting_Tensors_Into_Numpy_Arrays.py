import torch

x = torch.rand(2, 2)
print(x)
print(x.dtype)

y = x.numpy()
print(y)
print(y.dtype)