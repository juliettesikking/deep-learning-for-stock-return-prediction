import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train_model(model, data_loader, optimizer, epochs=10):
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Average Epoch Loss: {loss / len(data_loader)}")
    return loss.item()

# Generate synthetic data
input_size = 256
inputs = torch.randn(1000, input_size)
targets = torch.randn(1000, 1)
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=50)

# Set learning rates and model widths
learning_rates = np.logspace(-20, -10, num=10, base=2)
widths = [128, 256, 512, 1024]

# Plotting setup
plt.figure(figsize=(10, 5))
for width in widths:
    model = NeuralNetwork(input_size)
    losses = []
    for lr in learning_rates:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss = train_model(model, data_loader, optimizer, epochs=50)
        losses.append(loss)
    plt.plot(np.log2(learning_rates), losses, label=f'Width {width}')

plt.xlabel('log2(Learning Rate)')
plt.ylabel('Training Loss')
plt.title('Maximal Update Parametrization')
plt.legend()
plt.grid(True)
plt.show()
