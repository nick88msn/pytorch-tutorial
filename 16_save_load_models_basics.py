import torch
import torch.nn as nn

PATH = ""

### MAIN METHODS
# Saving a model
# we have two options, .save() saves the entire model
# Option 1: Lazy loading
torch.save(arg, PATH)              # Save not only models but also tensors or parameters as an argument, uses pickle to serialize objects
model = torch.load(PATH)           # Load the entire model that we have saved
model.eval()                       # Set the model to evaluation method

# Option 2: Reccomended way, saving and loading only the state dictionary of the model
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()