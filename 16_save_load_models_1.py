# Lazy Loading method
import torch
import torch.nn as nn

import os

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# model = Model(n_input_features=6)

# Saving
PATH = "./models/"
FILE = "model.pth"
# torch.save(model, os.path.join(PATH,FILE))

# Loading
model = torch.load(os.path.join(PATH, FILE))       # Load takes the filepath as argument
model.eval()

for param in model.parameters():
    print(param)