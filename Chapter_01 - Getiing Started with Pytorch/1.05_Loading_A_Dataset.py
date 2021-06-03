import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, utils

# path to store data and/or load from
path2data = "./data"

# loading training data
train_data = datasets.MNIST(path2data, train=True, download=True)

# extract training data and targets
x_train, y_train = train_data.data, train_data.targets
print(x_train.shape)
print(y_train.shape)

# loading validation data
val_data = datasets.MNIST(path2data, train=False, download=True)

# extract validation data and targets
x_val, y_val = val_data.data, val_data.targets
print(x_val.shape)
print(y_val.shape)

# add a dimension to tensor to become B*C*H*W
if len(x_train.shape) == 3:
    x_train = x_train.unsqueeze(1)
print(x_train.shape)

if len(x_val.shape) == 3:
    x_val = x_val.unsqueeze(1)
print(x_val.shape)

def show(img):
    # convert tensor to numpy array
    npimg = img.numpy()
    # Convert to H*W*C shape
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr, interpolation='nearest')
    plt.show()

# make a grid of 40 images, 8 images per row
x_grid = utils.make_grid(x_train[:40], nrow=8, padding=2)
print(x_grid.shape)
show(x_grid)
