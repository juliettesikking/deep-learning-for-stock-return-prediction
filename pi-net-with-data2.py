import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from experiments.networks.networks import PiNet

class PiCompleteModel(PiNet):
    def __init__(self, input_dim, hidden_layers, device="cpu"):
        super(PiCompleteModel, self).__init__()
        self.layers = [InfPiInputLinearReLU(input_dim, hidden_layers[0], device=device)]
        self.layers.extend([
            InfPiLinearReLU(hidden_layers[i], hidden_layers[i + 1], device=device)
            for i in range(len(hidden_layers) - 1)
        ])
        self.layers.append(InfPiLinearReLU(hidden_layers[-1], 1, output_layer=True, device=device))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

def load_data(filepath):
    data = pd.read_pickle(filepath)
    data['month'] = data['date'].dt.month + data['date'].dt.year * 100
    return data

def normalize_data(data, feature_cols):
    # Normalize features for better stability in training
    for col in feature_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std if std != 0 else data[col]
    return data

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return float('nan')
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return if std_return != 0 else float('nan')

def main(filepath: str, output_dir: str):
    data = load_data(filepath)
    feature_cols = [col for col in data.columns if col not in ['id', 'date', 'size_grp', 'r_1', 'month']]
    data = normalize_data(data, feature_cols)
    target_col = 'r_1'

    sharpe_ratios = []
    learning_rates = []

    for lr in np.logspace(-4, -2, num=10):
        learning_rate = lr
        model = PiCompleteModel(input_dim=len(feature_cols), hidden_layers=[16, 16, 16], device="cpu")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()  # Assuming MSRR is similar to MSE

        # Create a DataLoader instance
        features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
        targets = torch.tensor(data[target_col].values, dtype=torch.float32)
        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(10):  # Training epochs
            model.train()
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

        # Evaluate Sharpe ratio
        model.eval()
        with torch.no_grad():
            predictions = model(features)
            returns = predictions.squeeze().numpy() * np.random.randn(*predictions.squeeze().numpy().shape)  # Simulated returns
            sharpe_ratio = calculate_sharpe_ratio(returns)
            sharpe_ratios.append(sharpe_ratio)
            learning_rates.append(learning_rate)

        print("Learning Rates:", learning_rates)
        print("Sharpe Ratios:", sharpe_ratios)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates, sharpe_ratios, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio vs Learning Rate')
    plt.grid(True)
    plt.show()  # This will display the plot directly instead of saving

if __name__ == "__main__":
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle')
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()
    main(filepath=args.filepath,output_dir=args.output_dir)
