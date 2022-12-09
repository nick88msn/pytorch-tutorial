from torch_check import machine_check_setup
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = machine_check_setup()

# Hyper-parameters
#input_size = 784 # 28x28
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = .001

input_size = 28
sequence_length = 28
num_layers = 2      # Stacking 2 RNN with the 2nd taking the output of the firs as an input

# MNIST-dataset
train_dataset = torchvision.datasets.MNIST(root="./data/MNIST", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data/MNIST", train=False, transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # x -> batch_size, sequ_len, input_size
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        # out: batch_size, seq_len, hidden_size
        # out (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        out = self.fc(out)
        return out


model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28,28]
        # resized: [100, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels= labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1} / {num_epochs}\tStep {i+1} / {n_total_steps}\tLoss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max return value, index
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network on the 10000 test images: {acc} %")