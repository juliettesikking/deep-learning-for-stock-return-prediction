import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# Load the dataset from the pickle file
def load_dataset(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The dataset is not a pandas DataFrame.")
    return data


# Define the shallow neural network
class ShallowNet(nn.Module):
    def __init__(self, input_size):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, 1, bias=False)  # Output layer, no bias

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


# Assume the path to your pickle file is correct
pickle_file_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'
data = load_dataset(pickle_file_path)

# Preprocess the data
# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train.astype('float32'))
y_train_tensor = torch.tensor(y_train.astype('float32'))
X_test_tensor = torch.tensor(X_test.astype('float32'))
y_test_tensor = torch.tensor(y_test.astype('float32'))

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = ShallowNet(input_size=X_train.shape[1])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Evaluate on test data
model.eval()  # set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = outputs.squeeze()
        # Here you might apply a threshold or round off to get the prediction
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Accuracy on test set: {accuracy:.2f}')

# Predicting the next month's return
# Here we should use new data, but for the demonstration, let's use the test set
next_month_prediction = model(X_test_tensor)
print(f'Predictions for the next month: {next_month_prediction.squeeze().tolist()}')
