#### Training Pipeline
# 1) Design model (input size, output size, forward pass with all operations and layers)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass: compute prediction
#    - backward pass: gradients
#    - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

import time

# 0) Data Preparation
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

### Linear model expect batch_size and features as a vector column, so one column with batch_size row
### we need to convert
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
input_size = n_features
output_size = 1

### nn.Linear(input_size, output_size) => y = X transpose(w) + b; 
### with w a vector of size (1, n_features) and X a Tensor of size (n_samples, n_features)
### bias has size of output_size like y (output features)
### y has size output_size

model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 3) Training Loop
num_epoch = 100

start = time.perf_counter()
for epoch in range(num_epoch):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    # backward pass
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()

    [w,b] = model.parameters()

    if (epoch+1) % int(num_epoch/10) == 0:
        print(f"Epoch: {epoch + 1}\tWeights: {w.item():.3f}\tLoss:{loss.item():.8f}\tBias:{b.item():.3f}\tEpoch time:{(time.perf_counter() - start):.8f}")

# Plotting
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()