import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class InfMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers_count):
        super(InfMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers_count)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


def load_data(filepath):
    data = pd.read_pickle(filepath)
    data["month"] = data["date"].dt.month + data["date"].dt.year * 100
    feature_cols = [col for col in data.columns if col not in ["id", "date", "size_grp", "r_1", "month"]]
    features = data[feature_cols].values
    targets = data['r_1'].values  # Assuming 'r_1' is the target variable
    return features, targets


def calculate_sharpe_ratio(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return * np.sqrt(12)


def train_model(model, criterion, optimizer, features, targets, epochs=10):
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


def main(filepath, hidden_dims, layers_count):
    features, targets = load_data(filepath)
    plt.figure(figsize=(10, 5))

    for hidden_dim in hidden_dims:
        model = InfMLP(input_dim=features.shape[1], hidden_dim=hidden_dim, output_dim=1, layers_count=layers_count)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        train_model(model, criterion, optimizer, features, targets, epochs=50)

        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(features, dtype=torch.float32))
        sharpe_ratio = calculate_sharpe_ratio(predictions.numpy().flatten())

        plt.plot(hidden_dim, sharpe_ratio, marker="o", label=f"Hidden Dim: {hidden_dim}")

    plt.xlabel("Hidden Dimensions")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe Ratio vs Hidden Dimensions")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle")
    args = parser.parse_args()
    hidden_dims = [16, 64, 128]  # Example hidden dimensions
    layers_count = 2  # Number of hidden layers
    main(filepath=args.filepath, hidden_dims=hidden_dims, layers_count=layers_count)
