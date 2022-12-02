import torch
import torch.nn as nn
import numpy

loss = nn.CrossEntropyLoss()

### Careful! nn.CrossEntropyLoss already apply Softmax
### nn.CrossEntropyLoss = nn.LogSftmax + nn.NLLLoss (negative log likelihood loss)
### DO NOT ADD SOFTMAX IN LAST LAYER!!!

### Y has class labels, not One-Hot encoding
### Y_pred has raw scores (logits), no Softmax!

Y = torch.tensor([0])

# n_samples x nclasses = 1 x 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[.1, 3.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, dim=1)
_, predictions2 = torch.max(Y_pred_bad, dim=1)

print(predictions1, predictions2)

### If we have multiple samples
# 3 samples
Y = torch.tensor([0, 2, 1])

# n_samples x nclasses = 3 x 3
Y_pred_good = torch.tensor([
    [2.0, 1.0, 0.1],
    [.3, 1.0, 1.5],
    [1, .9, 0.3]
])
Y_pred_bad = torch.tensor([
    [.3, 2.0, 1.4],
    [1.3, .8, .3],
    [.1, .1, 0.8]
])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, dim=1)
_, predictions2 = torch.max(Y_pred_bad, dim=1)

print(predictions1, predictions2)