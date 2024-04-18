import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Function to load a pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Load your data
pickle_file_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'
data = load_pickle(pickle_file_path)

# Assuming the dataset structure, adjust as necessary
X = torch.randn(1000, 10)  # Random data for example purposes
y = torch.randint(0, 2, (1000,)).type(torch.LongTensor)  # Random binary labels

# Create TensorDataset and DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Neural network architecture for classification
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ShallowNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 2, bias=False)  # Output layer for 2 classes

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = self.layer2(x)
        return x

# Function to calculate accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Generate logarithmically spaced learning rates from 1e-6 to 1
learning_rates = np.logspace(-4, -1, num=200)

# Results storage
accuracies = []

# Iterate over each learning rate, train and calculate accuracy
for lr in learning_rates:
    model = ShallowNeuralNetwork(input_dim=X.size(1))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(100):  # You may adjust the number of epochs
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f'Learning Rate: {lr:.6f} - Accuracy: {accuracy:.2f}%')

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.semilogx(learning_rates, accuracies, marker='o', linestyle='-')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Learning Rate')
plt.grid(True)
plt.show()
