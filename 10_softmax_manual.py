import torch
import torch.nn
import numpy as np

### Calculate softmax manually using numpy
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(f'softmax numpy: {outputs}')

### Calculate softmax using pytorch built-in
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)