import numpy as np
import torch

x = np.zeros((2, 2), dtype=np.float32)
print(x)
print(x.dtype)

y = torch.from_numpy(x)
print(y)
print(y.dtype)