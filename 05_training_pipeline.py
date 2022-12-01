#### Training Pipeline
# 1) Design model (input size, output size, forward pass with all operations and layers)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass: compute prediction
#    - backward pass: gradients
#    - update weights

import torch
import torch.nn as nn
import time

### f = w * x

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

'''
# nn.Linear is a ready architecture, if we want a custom one we can implement as follows
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''

print(f"Prediction before training: f(5) = {model(x_test).item():.3f}")

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

start = time.perf_counter() 
for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    # loss
    l = loss(Y, y_pred)
    # gradients = backward
    l.backward()    # dl/dw
    # update weights
    optimizer.step()
    # zero gradients to avoid accumolation in w.grad
    optimizer.zero_grad()
    if epoch % int(n_iters/10) == 0:
        print(f"Epoch: {epoch + 1}\tWeights: {model.weight.item():.3f}\tLoss:{l:.8f}\tBias:{model.bias.item():.3f}\tEpoch time:{(time.perf_counter() - start):.8f}")


print(f"Prediction before training: f(5) = {model(x_test).item():.3f}\tTraining time:{(time.perf_counter() - start):.8f}")