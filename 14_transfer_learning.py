### From: https://www.kaggle.com/code/mnagao/pytorch-hymenoptera/notebook
# New concepts:
# ImageFolder -> Dataset imagefolder
# Scheduler -> to change the Learning Rate
# Transfer Learning -> Resuse a previously trained model changing the classification layer for new tasks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torch_check import machine_check_setup

device = 'cpu'      # not using mps cause it does not support Float64 and need to convert to Float32


mean = np.array([.485, .456, .406])
std = np.array([.229, .224, .225])

# Image visualizer helper function
plt.ion()   # interactive mode
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]), 'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# import data
data_dir = "./data/hymenoptera_data"
sets = ['train', 'val']
image_datasets = { x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets }
dataloaders = { x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in sets }

dataset_sizes = { x: len(image_datasets[x]) for x in sets }
class_names = image_datasets['train'].classes
print(class_names)

'''
# Let's visualize a few training images so as to understand the data augmentations.
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
'''

'''
Training the model
Now, let's write a general function to train a model. Here, we will illustrate:

Scheduling the learning rate
Saving the best model
In the following, parameter scheduler is an LR scheduler object from torch.optim.lr_scheduler.
'''

# Hyper-parameters
num_epochs = 25

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.perf_counter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"{epoch + 1} / {num_epochs}")
        print('-'*10)

        for phase in sets:
            if phase == 'train':
                model.train()   # sets model to training mode
            elif phase == 'val':
                model.eval()
            
            running_loss = .0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled( phase == "train" ):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum( preds == labels.data )
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = 100.0 * (running_corrects.double() / dataset_sizes[phase])

            print(f"{phase} Loss:{epoch_loss}\tAcc: {epoch_acc:.3f}%") 

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.perf_counter() - since
    print(f"Training complete in {time_elapsed / 60:.0f}mins and {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.3f}%")

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model

'''
Finetuning the convnet
Load a pretrained model and reset final fully connected layer.

ConvNet as fixed feature extractor
Here, we need to freeze all the network except the final layer. We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().

You can read more about this in the documentation here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>__.
'''

# Loading a pretrained model
model = models.resnet18(pretrained=True)    #pretrained deprecrated use weights=ResNet18_Weights.DEFAULT instead

# If we want to fix model parameters and do the gradient only on the last layer
for param in model.parameters():
    param.requires_grad = False

# Exchange last fully connected layers
num_ftrs = model.fc.in_features     #number of features in the last fully connected layer

# Create a new layer and assign to the model last layer
model.fc = nn.Linear(num_ftrs, 2)       # two classes, ants or bees
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.001)

# Scheduler -> will update lr 
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)       # every 7 epochs the learning rate is increased of 10%

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs)

'''
Visualizing the model predictions

Generic function to display predictions for a few images
'''

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(model)

plt.ioff()
plt.show()