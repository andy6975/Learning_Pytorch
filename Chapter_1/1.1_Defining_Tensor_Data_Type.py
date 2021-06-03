import torch

x = torch.ones(2, 2)
print(x)
print(x.dtype)

x = torch.ones(2, 2, dtype=torch.int8)
print(x)
print(x.dtype)

x = torch.ones(1, dtype=torch.uint8)
x = x.type(torch.float)
print(x)
print(x.dtype)
