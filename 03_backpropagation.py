import torch

from torch_check import machine_check_setup

device = machine_check_setup()

# Steps
# 1. Forward pass: Compute Loss
# 2. Compute local gradients
# 3. Backward pass: Compute dLoss / dWeights using the Chain Rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#forward pass
y_hat = w * x
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)

# update weights
# next forward and backward