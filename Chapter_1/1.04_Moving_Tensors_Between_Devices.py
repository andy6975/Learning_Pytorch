import torch

x = torch.tensor([1.5, 2])
print(x)
print(x.dtype)
print(x.device)

if torch.cuda.is_available():
    device = torch.device("cuda:0")

x = x.to(device)
print(x)
print(x.dtype)
print(x.device)

device = torch.device("cpu")
x = x.to(device)
print(x)
print(x.dtype)
print(x.device)

device = torch.device("cuda:0")
y = torch.ones(2, 2, device=device)
print(y)
print(y.dtype)
print(y.device)
