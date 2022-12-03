'''
Activation function
Activation functions apply a non-linear transformation and decide wheter a neuron should be activated or not.
Without activation functions our network is basically just a stacked linear regression model and cannot perform more complex tasks
After each layer we tipically want to apply an activation function
Most popular activation functions:
    1) Step function (threshold z-shape function, not used in practice)
    2) Sigmoid  (0-1 tipically in last layer of a binary classification problem to get a probability)
    3) Tanh     (scaled sigmoid function -1 - 1 usually in hidden layers)
    4) ReLU     (most popular choice, 0 for negative values, input as output for positive values) f(x) = max(0,x)
    5) Leaky ReLU (slightly modified ReLU with negative values multiplied by a very small number like .001, tries to resolved the Vanishing Gradient problem)
    6) Softmax  (probability as an output, good choice for last layer of multi class classification problems)
'''

import torch
import torch.nn as nn
import torch.nn.function as F

# Option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# Option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out

