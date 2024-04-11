import torch
import torch.nn as nn
import pickle
import pandas as pd

# Load the dataset from the pickle file
def load_dataset(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    # If data is a pandas DataFrame, you can get the number of features like this:
    if isinstance(data, pd.DataFrame):
        # Assuming the last column is the target variable
        input_size = data.shape[1] - 1
        features = data.iloc[:, :-1]  # all columns except the last one
        target = data.iloc[:, -1]     # the last column as target
    else:
        raise ValueError("The dataset is not in a recognized format. Please ensure it is a pandas DataFrame.")
    return features, target, input_size

# Replace with the path to your .pickle file
pickle_file_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'

# Load your dataset and get the number of features
features, target, input_size = load_dataset(pickle_file_path)

# Define the shallow neural network
class ShallowNet(nn.Module):
    def __init__(self, input_size):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, 1, bias=False)  # Output layer, no bias

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)  # No activation function for output layer
        return x

# Initialize the model
model = ShallowNet(input_size=input_size)

# Here you will add your code for the training loop.
# You will train the model using your `features` and `target`.
