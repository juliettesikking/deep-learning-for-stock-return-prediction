import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset

# Function to load a pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# Replace 'your_pickle_file_path.pickle' with your actual file path
pickle_file_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'
data = load_pickle(pickle_file_path)

# Assuming 'data' is a dictionary with 'features' and 'targets'
# Preprocess your features and targets as needed here
# X = preprocess_features(data['features'])
# y = preprocess_targets(data['targets'])

# For the sake of the example, let's create dummy tensors
# You should replace these lines with actual preprocessing
X = torch.randn(1000, 10)  # Example feature tensor with 1000 samples and 10 features each
y = torch.randn(1000, 1)   # Example target tensor with 1000 samples and 1 target each

# Create TensorDataset and DataLoader
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
input_dim = X.size(1)  # Use the second dimension of X as the input dimension
model = ShallowNeuralNetwork(input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 200  # Placeholder for the number of epochs
for epoch in range(epochs):
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs or according to your preference
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Save the model after training
torch.save(model.state_dict(), 'shallow_network_model.pth')

print("Training complete.")
