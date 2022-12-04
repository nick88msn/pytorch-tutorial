# pip install tensorboard
# https://pytorch.org/docs/stable/tensorboard.html

from torch_check import machine_check_setup
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import time, sys

writer = SummaryWriter("runs/mnist")

# Device configuration
device = machine_check_setup()

# Hyper parameters
input_size = 28*28  # 28px * 28px = 784
hidden_size = 100
num_classes = 10    # Digits from 0 to 9
num_epochs = 10
batch_size = 100
learning_rate = .01

# MNIST data import
train_dataset = torchvision.datasets.MNIST(root='./data/MNIST/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)


# Get and display a sample of the dataset
examples = iter(train_loader)
samples, labels = next(examples)

print(samples.shape, labels.shape)

'''
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
'''

# Write the sample into tensorboard
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# writing the computational graph into tensorboard
writer.add_graph(model, samples.to(device).reshape(-1, 28*28))

# Training Loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0

global_start = time.perf_counter()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        iter_start = time.perf_counter()
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging to tensorboard
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        running_correct += ( predicted == labels ).sum().item()


        if (i+1) % int(n_total_steps / 100):
            print(f"Epoch:{epoch}/{num_epochs}\tStep:{i}/{n_total_steps}\tLoss:{loss.item()}\tIter:{time.perf_counter() - iter_start:.4f}s")
            writer.add_scalar('training loss', running_loss/int(n_total_steps / 100), epoch *  n_total_steps + i)
            writer.add_scalar('training accuracy', running_correct/int(n_total_steps / 100), epoch *  n_total_steps + i)
            running_loss = .0
            running_correct = 0

# Perf measures
global_end = time.perf_counter() - global_start
print(f"Training Time: {global_end:.4f}s\tTraining per epoch: {global_end/num_epochs:.4}s")

# Testing model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * (n_correct/n_samples)
    print(f"Model Accuracy: {acc}%")

writer.close()
