import argparse
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from losses.MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from pilim.experiments.networks.networks import PiNet


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

        self.d_in = d_in
        self.d_out = d_out
        self.r = r
        self.L = L

        self.register_param_buffer("first_layer_alpha", first_layer_alpha)
        self.register_param_buffer("last_layer_alpha", last_layer_alpha)
        self.register_param_buffer("bias_alpha", bias_alpha)
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_param_buffer("last_bias_alpha", last_bias_alpha)
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device))
        for n in range(1, L + 1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm,
                                               cuda_batch_size=cuda_batch_size))

        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device,
                                           layernorm=layernorm, cuda_batch_size=cuda_batch_size))

        self.initialize_weights()

    def register_param_buffer(self, param_name, value):
        # Register individual floats as buffers for later saving/loading
        self.register_buffer(param_name, torch.tensor(value, dtype=torch.get_default_dtype()))

    def initialize_weights(self):
        with torch.no_grad():
            for n, layer in enumerate(self.layers):
                if isinstance(layer, InfPiInputLinearReLU) or isinstance(layer, InfPiLinearReLU):
                    if hasattr(layer, 'weight'):
                        if n == 0:
                            # Initialize first layer weights
                            nn.init.normal_(layer.weight, mean=0, std=1 / np.sqrt(self.d_in))
                        elif n == self.L + 1:
                            # Initialize last layer weights
                            nn.init.normal_(layer.weight, mean=0, std=1 / np.sqrt(self.r))
                        else:
                            # Initialize hidden layer weights
                            nn.init.normal_(layer.weight, mean=0, std=1 / np.sqrt(self.r))

    def forward(self, x):
        for n in range(0, self.L + 2):
            x = self.layers[n](x)
            print(f"Layer {n} output shape: {x.shape}")  # Debugging print to check shapes
            if n == 0:
                x *= self.first_layer_alpha
            if n == self.L + 1:
                x *= self.last_layer_alpha
        return x


def load_data(filepath):
    data = pd.read_pickle(filepath)
    data["month"] = data["date"].dt.month + data["date"].dt.year * 100
    return data


def calculate_sharpe_ratio(returns):
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return * np.sqrt(12)


def run_experiment(hidden_dim, data, feature_cols, window_size):
    sharpe_ratios = []
    learning_rates = []
    for lr in np.arange(-18, -9, step=1):
        model = InfMLP(
            d_in=len(feature_cols),
            d_out=1,
            r=hidden_dim,
            L=2,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        optimizer = optim.Adam(model.parameters(), lr=2.0 ** lr)
        criterion = MSRR()
        rolling_window = RollingWindow(data, window_size=window_size)
        torch_runner = TorchRunner(
            rolling_window=True,
            window_size=window_size,
            end_year=2023,
            output="./",
            epochs=10,
            data=data,
            resume=False,
            no_incremental=False,
            model_name=model,
            coordinate_check=False,
        )
        returns = torch_runner.run(model, criterion, optimizer, rolling_window)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        sharpe_ratios.append(sharpe_ratio)
        learning_rates.append(lr)
    return learning_rates, sharpe_ratios


def main(filepath: str, window_size: int):
    data = load_data(filepath)
    feature_cols = [
        col
        for col in data.columns
        if col not in ["id", "date", "size_grp", "r_1", "month"]
    ]
    network_sizes = [16, 64, 128]
    plt.figure(figsize=(10, 5))

    for size in network_sizes:
        learning_rates, sharpe_ratios = run_experiment(
            size, data, feature_cols, window_size
        )
        plt.plot(
            learning_rates, sharpe_ratios, marker="o", label=f"Network size: {size}"
        )

    plt.xlabel("Learning Rate ")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe Ratio vs Learning Rate")
    ax = plt.gca()
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        default="/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle",
    )
    args = parser.parse_args()

    # 5 years = 60 months
    main(filepath=args.filepath, window_size=5)
