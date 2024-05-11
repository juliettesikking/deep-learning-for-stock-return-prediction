import argparse
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from experiments.networks.networks import PiNet

class InfMLP(PiNet):
    def __init__(
            self,
            d_in,
            d_out,
            r,
            L,
            first_layer_alpha=1,
            last_layer_alpha=1,
            bias_alpha=1,
            last_bias_alpha=None,
            layernorm=False,
            cuda_batch_size=None,
            device="cpu"):
        super(InfMLP, self).__init__()

        # Register buffers correctly
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.get_default_dtype()))
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.get_default_dtype()))

        self.layers = nn.ModuleList()
        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device))
        for _ in range(1, L):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size))
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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

def run_experiment(network_config, data, feature_cols):
    sharpe_ratios = []
    learning_rates = []
    for lr in np.logspace(-4, -2, num=10):
        model = InfMLP(d_in=len(feature_cols), d_out=1, r=network_config['r'], L=network_config['L'], device='cpu')
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
    network_configs = [{'r': 16, 'L': 3}, {'r': 64, 'L': 3}, {'r': 128, 'L': 3}]
    plt.figure(figsize=(10, 5))

    for config in network_configs:
        learning_rates, sharpe_ratios = run_experiment(config, data, feature_cols)
        plt.plot(learning_rates, sharpe_ratios, marker='o', label=f'Layers config: r={config['r']}, L={config['L']}')

    plt.xscale('log', base=2)  # Correctly set the log base to 2
    plt.xlabel('Learning Rate (log scale base 2)')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio vs Learning Rate using Pi Network')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str,
                        default='/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle')
    args = parser.parse_args()
    main(filepath=args.filepath)
