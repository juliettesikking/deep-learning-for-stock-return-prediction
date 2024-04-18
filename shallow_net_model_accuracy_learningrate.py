import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

# Function to load a pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Load your data
pickle_file_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'
data = load_pickle(pickle_file_path)

# Assume data is preprocessed for classification
X = torch.randn(1000, 10)  # Random data for example purposes, replace with your data
y = torch.randint(0, 2, (1000, 1)).squeeze()  # Random binary labels for example purposes, replace with your data

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

# Initialize the neural network
input_dim = X.size(1)  # Number of input features
model = ShallowNeuralNetwork(input_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)  # Reduces the learning rate by a factor of 10 every 25 epochs

# Function to calculate accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Training loop
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    total_acc = 0
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = accuracy(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    # Update learning rate
    scheduler.step()

    # Print average loss and accuracy
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

# Save the model
torch.save(model.state_dict(), 'shallow_network_model.pth')

print("Training complete.")
