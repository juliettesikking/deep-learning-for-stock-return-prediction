import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Transform the data to torch tensors and normalize it
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download the training and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Loaders that will handle batching
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define a simplified MLP model without the fc2 layer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        # Removed the fc2 layer
        self.fc3 = nn.Linear(512, 10)  # Adjusted to take input directly from fc1
        self.relu = nn.ReLU()

    def forward(self, x):        x = x.view(-1, 28 * 28)  # Flatten the images
        x = self.relu(self.fc1(x))
        # Skipped processing with fc2
        x = self.fc3(x)  # Directly from fc1 to fc3
        return x

# Function to train and evaluate the model
def train_and_evaluate_model(learning_rate):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(10):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Main script
# Generate 20 logarithmically spaced learning rates between 1e-4 and 1e-1
learning_rates = np.logspace(-4, -1, num=20)
accuracies = []

for lr in learning_rates:
    accuracy = train_and_evaluate_model(lr)
    accuracies.append(accuracy)
    print(f'Learning Rate: {lr:.4f} - Accuracy: {accuracy:.2f}%')

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, accuracies, '-o')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Learning Rate')
plt.grid(True)
plt.show()
