import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from torch_check import machine_check_setup

device = 'cpu'

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

images, labels = next(iter(data_loader))

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super(AutoencoderCNN, self).__init__()
        # N, 28, 28 (batch_size, 28, 28)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16 , 3, stride=2, padding=1),  # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32 , 3, stride=2, padding=1), # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64 , 7),   # N, 64, 1 , 1
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),   # N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # N, 1, 28, 28
            nn.Sigmoid()                         # returns value into (0,1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Note: if input [-1, 1] -> we do not need Sigmoid and we use Tanh function

model = AutoencoderCNN().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)

# Train

num_epochs = 5
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}\tLoss: {loss.item():.4f}")
    outputs.append((epoch, img, recon))

# Reconstruct the images
for k in range(0, num_epochs, 1):
    plt.figure(figsize=(9,2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i , item in enumerate(imgs):
        if i>= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])
    for i , item in enumerate(recon):
        if i>= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])
plt.show()