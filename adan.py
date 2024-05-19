import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28 images
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# MNIST dataset
train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Calculate the number of iterations per epoch and determine plot interval
total_iterations_per_epoch = len(train_loader)
desired_plot_points = 200
plot_every = max(1, int(total_iterations_per_epoch * num_epochs / desired_plot_points))


# Neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x.view(-1, input_size))
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Function to apply a simple moving average for smoothing the loss curves
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# Training function
def train(optimizer_name):
    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_dict = {
        'Adam': optim.Adam(model.parameters(), lr=learning_rate),
        'AdaGrad': optim.Adagrad(model.parameters(), lr=learning_rate),
        'RMSProp': optim.RMSprop(model.parameters(), lr=learning_rate),
        'SGDNesterov': optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True),
        'AdaDelta': optim.Adadelta(model.parameters(), lr=learning_rate)
    }
    optimizer = optimizer_dict[optimizer_name]

    # List to store loss to plot
    losses = []

    # Train the model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss at specified intervals
            if (epoch * total_iterations_per_epoch + i) % plot_every == 0:
                losses.append(loss.item())

    # Apply smoothing to the recorded losses
    smoothed_losses = smooth_curve(losses)
    return smoothed_losses


# Running the experiment
optimizers = ['Adam', 'AdaGrad', 'RMSProp', 'SGDNesterov', 'AdaDelta']
losses_dict = {}

for opt in optimizers:
    losses = train(opt)
    losses_dict[opt] = losses
    plt.plot(losses, label=opt, linewidth=1)  # Thinner line with smoothing

plt.title('Optimizer Comparison on MNIST')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()
