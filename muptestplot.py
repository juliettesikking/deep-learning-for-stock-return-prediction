import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

class VariableComplexityModel(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units):
        super(VariableComplexityModel, self).__init__()
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.ReLU()])
        layers.append(nn.Linear(hidden_units, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, data_loader, optimizer, epochs=50):
    criterion = nn.MSELoss()
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(loss)
    return total_loss / (len(data_loader) * epochs)

def generate_data(samples=1000, features=256):
    inputs = torch.randn(samples, features)
    targets = torch.randn(samples, 1)
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=50)
    return data_loader

def run_experiment():
    input_size = 256
    learning_rates = np.logspace(-20, -10, num=10, base=2)  # Log2 scale from -20 to -10
    complexities = [(1, 256), (2, 128), (3, 64)]  # Pairs of (hidden_layers, hidden_units)

    plt.figure(figsize=(10, 6))
    for hidden_layers, hidden_units in complexities:
        model = VariableComplexityModel(input_size, hidden_layers, hidden_units)
        data_loader = generate_data()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        losses = []
        for lr in learning_rates:
            optimizer.param_groups[0]['lr'] = lr
            loss = train_model(model, data_loader, optimizer, epochs=50)
            losses.append(loss)
        plt.plot(np.log2(learning_rates), losses, label=f'Layers: {hidden_layers}, Units: {hidden_units}')

    plt.xlabel('log2(Learning Rate)')
    plt.ylabel('Training Loss')
    plt.title('Maximal Update Parametrization with Variable Model Complexity')
    plt.legend()
    plt.grid(True)
    plt.show()

run_experiment()
