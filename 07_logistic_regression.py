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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import time

# 0) Data Preparation
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Converting from double to float32 for better compatibility with torch Tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshaping data to vector columns
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model
# f = wx + b , sigmoid at the end

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)
# 2) Loss and optimizer
learning_rate = .01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 3) Training Loop
num_epochs = 10000

start = time.perf_counter()
for epoch in range(num_epochs):
    # forward and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if (epoch+1) % int(num_epochs/10) == 0:
        print(f"Epoch: {epoch + 1}\tLoss:{loss.item():.8f}\tEpoch time:{(time.perf_counter() - start):.8f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])      #for every correct prediction it will sum 1
    print(f"Accuracy = {acc:.4f}")