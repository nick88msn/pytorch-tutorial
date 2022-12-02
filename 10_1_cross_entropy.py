### Used in multi-class problems
### measure the loss amplifying the distance
### D(y_predicted, y) = -1/N * sum(yi * log(y_predicted))

### y has to be one-hot encoded, so usually a softmax output is like this y_predicted = [.7,.2,.01]
### while cross-entropy needs labels like y_predicted = [1,0,0]

import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = - np.sum(actual * np.log(predicted))
    return loss     # /float(predicted.shape[0])

# y meust be one hot encoded
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]
Y = np.array([1,0,0])

# y_pred has probabilities
Y_pred_good = np.array([.7,.3,.1])
Y_pred_bad = np.array([.1,.3,.5])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f"Loss1 numpy:{l1}")
print(f"Loss2 numpy:{l2}")