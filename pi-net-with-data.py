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
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from experiments.networks.networks import PiNet

class PiCompleteModel(PiNet):  # Changed from torch.nn.Module to PiNet
    def __init__(self, input_dim, hidden_layers, device="cpu"):
        super(PiCompleteModel, self).__init__()
        self.linear = InfPiInputLinearReLU(input_dim, hidden_layers[0], device=device)
        self.hidden_layers = torch.nn.ModuleList([
            InfPiLinearReLU(hidden_layers[i], hidden_layers[i + 1], device=device)
            for i in range(len(hidden_layers) - 1)
        ])
        self.output_layer = InfPiLinearReLU(hidden_layers[-1], 1, output_layer=True, device=device)  # Changed to InfPiLinearReLU with output_layer=True
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x):
        x = self.non_linearity(self.linear(x))
        for hidden_layer in self.hidden_layers:
            x = self.non_linearity(hidden_layer(x))
        x = self.output_layer(x)
        return x

def load_data(filepath):
    data = pd.read_pickle(filepath)
    data['month'] = data['date'].dt.month + data['date'].dt.year * 100
    return data

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if returns is None or len(returns) == 0:
        return float('nan')  # Handle empty or None returns gracefully

    # Remove NaN values from returns
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return float('nan')  # Handle case where all returns were NaN

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return float('nan')  # Avoid division by zero if no variation in returns

    return (mean_return / std_return)

def main(filepath: str, output_dir: str):
    data = load_data(filepath)
    feature_cols = [col for col in data.columns if col not in ['id', 'date', 'size_grp', 'r_1', 'month']]
    target_col = 'r_1'

    sharpe_ratios = []
    learning_rates = []

    for lr in np.logspace(-4, -2, num=10):
        learning_rate = lr
        model = PiCompleteModel(input_dim=len(feature_cols), hidden_layers=[16, 16, 16], device="cpu")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = MSRR()

        rolling_window = RollingWindow(data, window_size=12)  # Window size adjusted here if needed
        torch_runner = TorchRunner(
            rolling_window=True,
            window_size=24,
            end_year=2023,
            output=output_dir,
            epochs=10,
            data=data,
            resume=False,
            no_incremental=False,
            model_name=model,
            coordinate_check=False
        )

        returns = torch_runner.run(model, criterion, optimizer, rolling_window)  # Assuming this returns the necessary returns
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
    plt.savefig(f"{output_dir}/sharpe_vs_lr.png")
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default='/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle')
    parser.add_argument("--output-dir", type=str, default="./")
    args = parser.parse_args()
    main(filepath=args.filepath, output_dir=args.output_dir)
