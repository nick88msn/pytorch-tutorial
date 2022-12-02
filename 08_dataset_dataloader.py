'''
epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in one forward & backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
'''

import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

### Create custom Dataset
class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])       # n_samples, 1
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]         # dataset[0]

    def __len__(self):
        return self.n_samples                       # len(dataset)

dataset = WineDataset()
batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# training loop
num_epochs = 10
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size)

print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1) % 5 == 0:
            print(f'Epoch:{epoch + 1}/{num_epochs}\tStep:{i}/{n_iterations}\tinputs{inputs.shape}')