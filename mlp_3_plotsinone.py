import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner


class CompleteModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(CompleteModel, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        # Input layer
        self.hidden_layers.append(torch.nn.Linear(input_dim, hidden_layers[0]))
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        # Output layer
        self.output_layer = torch.nn.Linear(hidden_layers[-1], 1, bias=False)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.non_linearity(layer(x))
        return self.output_layer(x)


def load_data(filepath):
    data = pd.read_pickle(filepath)
    data['month'] = data['date'].dt.month + data['date'].dt.year * 100
    return data


def calculate_sharpe_ratio(returns):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return float('nan')
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return if std_return != 0 else float('nan')


def run_experiment(hidden_layers, data, feature_cols):
    sharpe_ratios = []
    learning_rates = []
    for lr in np.logspace(-4, -2, num=10):
        model = CompleteModel(input_dim=len(feature_cols), hidden_layers=hidden_layers)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = MSRR()
        rolling_window = RollingWindow(data, window_size=12)
        torch_runner = TorchRunner(
            rolling_window=True,
            window_size=24,
            end_year=2023,
            output="./",
            epochs=10,
            data=data,
            resume=False,
            no_incremental=False,
            model_name=model,
            coordinate_check=False
        )
        returns = torch_runner.run(model, criterion, optimizer, rolling_window)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        sharpe_ratios.append(sharpe_ratio)
        learning_rates.append(lr)
    return learning_rates, sharpe_ratios


def main(filepath: str):
    data = load_data(filepath)
    feature_cols = [col for col in data.columns if col not in ['id', 'date', 'size_grp', 'r_1', 'month']]
    network_sizes = [[16, 16, 16], [64, 64, 64], [128, 128, 128]]
    plt.figure(figsize=(10, 5))

    for size in network_sizes:
        learning_rates, sharpe_ratios = run_experiment(size, data, feature_cols)
        plt.plot(learning_rates, sharpe_ratios, marker='o', label=f'Hidden layers: {size}')

    plt.xscale('log', base=2)  # Correctly set the log base to 2
    plt.xlabel('Learning Rate (log scale base 2)')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio vs Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        default='/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle')
    args = parser.parse_args()
    main(filepath=args.filepath)
