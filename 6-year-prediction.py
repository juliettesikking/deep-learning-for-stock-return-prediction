import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to load a pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Load and preprocess the data
pickle_file_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'
data = load_pickle(pickle_file_path)

# Remove columns not required for training
y = data.pop('r_1')  # Target: next month's return
data.pop('size_grp')
data.pop('id')

# Convert data into a DataFrame for preprocessing
df = pd.DataFrame(data)
df['r_1'] = y

# Convert all columns to numeric, replace non-numeric values with NaN and fill NaNs with the mean
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Convert DataFrame to PyTorch tensors
X = torch.tensor(df.drop('r_1', axis=1).values, dtype=torch.float32)
y = torch.tensor(df['r_1'].values, dtype=torch.float32).view(-1, 1)

# Define the dataset and dataloader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Neural network architecture
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ShallowNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = self.layer2(x)
        return x

# Initialize the neural network
model = ShallowNeuralNetwork(input_dim=X.shape[1])

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training function
def train_model(epochs, model, dataloader, optimizer, criterion):
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# Example training call
train_model(200, model, dataloader, optimizer, criterion)

# Save the model after training
torch.save(model.state_dict(), 'shallow_network_model.pth')
logging.info("Training complete.")

# Example prediction (you should replace this part with your prediction code)
model.eval()
with torch.no_grad():
    predictions = model(X)
    print(predictions)
