import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

path2data = "./data"

# loading MNIST training dataset
train_data = datasets.MNIST(path2data, train=True, download=True)

# define transformation
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor()
])

ind = np.random.randint(60000)

# get a sample image from training dataset
img = train_data[ind][0]

img_tr = data_transform(img)
img_tr = img_tr.numpy()

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(img_tr[0], cmap='gray')
plt.title("Transformed")
plt.show()

# we can also pass the transformer function to the dataset class
train_data = datasets.MNIST(path2data, train=True, download=True, transform=data_transform)

plt.subplot(1, 2, 1)
plt.imshow(train_data[ind][0][0], cmap='gray')
plt.title("First")
plt.subplot(1, 2, 2)
plt.imshow(train_data[ind-5][0][0], cmap='gray')
plt.title("Second")
plt.show()
