import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class MyModel(nn.Module):
    def __init__(self, width):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(784, width)
        self.output_layer = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the images
        x = torch.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x

def train_model(learning_rate, width=128):
    # Prepare MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = MyModel(width=width)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
    return average_loss

learning_rates = np.logspace(-20, -10, num=11, base=2)
losses = [train_model(lr) for lr in learning_rates]

plt.figure(figsize=(10, 5))
plt.plot(np.log2(learning_rates), losses, marker='o')
plt.xlabel('log2(Learning Rate)')
plt.ylabel('Training Loss')
plt.title('Maximal Update Parametrization with Adam Optimizer for Width 128 on MNIST')
plt.grid(True)
plt.show()
