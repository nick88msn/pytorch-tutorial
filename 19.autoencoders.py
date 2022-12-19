# Autoencoders
# Original Image -> Encoder -> Flatten Image -> Decoder -> Image reconstructed

# A possible application is video compression
# Instead of sending the whole image we send the encoded compressed one
# On the other side we have the decoder that reconstruct frames

# For both encoder and decoder we can use:
# a) A Feedforward Neural Net
# b) A Convolutional Neural Net (works better with images)

# Such models are "Generative models"
# Instead of doing a classification we generate image based on encoding

# To train our model we need a minimize function that we need to optimize
# We want our reconstructed image to be as close as the original one
# all pixel value should be the same so we apply the Mean Squared Error (MSE) 1/M sum(y - y_hat)^2

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from torch_check import machine_check_setup

device = "cpu"

transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

images, labels = next(iter(data_loader))

class AutoencoderLinear(nn.Module):
    def __init__(self):
        super(AutoencoderLinear, self).__init__()
        # N, 28 x 28 (batch_size, 784)
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),               # N, 784 -> N, 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)                    # N, 3
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),                    # N, 3
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),               # N, 784 -> N, 128
            nn.Sigmoid()                         # returns value into (0,1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Note: if input [-1, 1] -> we do not need Sigmoid and we use Tanh function

model = AutoencoderLinear().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)

# Train

num_epochs = 100
outputs = []
for epoch in range(num_epochs):
    for (img, _) in data_loader:
        img = img.reshape(-1, 28*28).to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}\tLoss: {loss.item():.4f}")
    outputs.append((epoch, img, recon))

# Reconstruct the images
for k in range(0, num_epochs, 10):
    plt.figure(figsize=(9,2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i , item in enumerate(imgs):
        if i>= 9: break
        plt.subplot(2, 9, i+1)
        item = item.reshape(-1, 28,28)
        plt.imshow(item[0])
    for i , item in enumerate(recon):
        if i>= 9: break
        plt.subplot(2, 9, 9+i+1)
        item = item.reshape(-1, 28,28)
        plt.imshow(item[0])
plt.show()