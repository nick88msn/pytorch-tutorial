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


model = Model(n_input_features=6)

# Saving Parameters
PATH = "./models/"
MODEL_FILE = "model.pth"
CHECKPOINT_FILE = "checkpoint.pth"

# Training Model
learning_rate = .01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model.state_dict())

# Saving Model
#torch.save(model.state_dict(), os.path.join(PATH,MODEL_FILE))

# Saving Checkpoint
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

torch.save(checkpoint, os.path.join(PATH,CHECKPOINT_FILE))

# Loading Checkpoint to continue training
loaded_checkpoint = torch.load(os.path.join(PATH,CHECKPOINT_FILE))
epoch = loaded_checkpoint["epoch"]
model_state = loaded_checkpoint["model_state"]

model.load_state_dict(model_state)
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

# Loading a model
#device = torch.device("mps")      # if we are using the GPU we need to specify it when loading the model
#loaded_model = Model(n_input_features=6)
#loaded_model.load_state_dict(torch.load(os.path.join(PATH,MODEL_FILE)), map_location=device)
#loaded_model.eval()

#for param in loaded_model.parameters():
#    print(param)