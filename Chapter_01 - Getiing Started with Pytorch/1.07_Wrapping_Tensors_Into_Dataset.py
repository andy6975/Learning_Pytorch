from torch.utils.data import TensorDataset
from torchvision import datasets

path2data = "./data"

# load MNIST data for example
train_data = datasets.MNIST(path2data, train=True, download=True)
val_data = datasets.MNIST(path2data, train=False, download=True)

x_train, y_train = train_data.data, train_data.targets
x_val, y_val = val_data.data, val_data.targets

# wrap tensors into a dataset
train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)

for x, y in train_ds:
    print(x.shape, y.item())
    break