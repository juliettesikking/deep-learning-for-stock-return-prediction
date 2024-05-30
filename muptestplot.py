import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# Define a simple model with a customizable width
class SimpleModel(nn.Module):
    def __init__(self, input_width, output_width=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_width, output_width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_width, 1)  # Output a single value

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model, data_loader, optimizer, epochs=5):
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        total_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Average Epoch Loss: {total_loss / len(data_loader)}")
    return total_loss / len(data_loader)

# Adjust the learning rate range and widths as needed
learning_rates = np.logspace(-20, -10, num=10, base=2)
widths = [128, 256, 512, 1024]  # Different widths to simulate

# Create a dataset and loader for each model width
results = {}
plt.figure(figsize=(10, 5))

for width in widths:
    # Generate data that matches the input width of the model
    inputs = torch.randn(100, width)  # 100 samples, 'width' features
    targets = torch.randn(100, 1)  # 100 target values
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=10)

    model = SimpleModel(input_width=width)
    optimizer = optim.Adam(model.parameters())

    losses = []
    for lr in learning_rates:
        optimizer.param_groups[0]['lr'] = lr
        loss = train_model(model, data_loader, optimizer, epochs=50)
        losses.append(loss)

    results[width] = losses
    plt.plot(np.log2(learning_rates), losses, label=f'Width {width}')

plt.xlabel('-log2(Learning Rate)')
plt.ylabel('Training Loss')
plt.title('Maximal Update Parametrization')
plt.legend()
plt.grid(True)
plt.show()
