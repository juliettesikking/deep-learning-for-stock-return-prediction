import os
import argparse
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from losses.MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner
import logging

logging.basicConfig(level=logging.INFO)

from torch import nn
from pilimit_lib.inf.layers import (
    InfPiInputLinearReLU,
    InfPiLinearReLU,
    FinPiLinearReLU,
)
from experiments.networks import PiNet
from pilimit_lib.inf.optim import PiSGD, store_pi_grad_norm_, clip_grad_norm_


class PiNetwork(PiNet):
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
        device="cpu",
    ):
        super(PiNetwork, self).__init__()

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

        self.layers.append(
            InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device)
        )
        for n in range(1, L + 1):
            self.layers.append(
                InfPiLinearReLU(
                    r,
                    device=device,
                    bias_alpha=bias_alpha,
                    layernorm=layernorm,
                    cuda_batch_size=cuda_batch_size,
                )
            )

        self.layers.append(torch.nn.Linear(r, 1, bias=False))

    def register_param_buffer(self, param_name, value):
        # Register individual floats as buffers for later saving/loading
        self.register_buffer(
            param_name, torch.tensor(value, dtype=torch.get_default_dtype())
        )

    def forward(self, x):
        for n in range(0, self.L + 2):
            x = self.layers[n](x)

            if n == 0:
                x *= self.first_layer_alpha
            if n == self.L + 1:
                x *= self.last_layer_alpha
        return x


class DeepNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, depth: int):
        super(DeepNeuralNetwork, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_size, bias=True)

        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size, bias=True) for i in range(depth)]
        )
        self.output_layer = torch.nn.Linear(hidden_size, 1, bias=False)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x):
        x = self.non_linearity(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.non_linearity(hidden_layer(x))

        return self.output_layer(x)

    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


class ShallowNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(ShallowNetwork, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = torch.nn.Linear(hidden_size, 1, bias=False)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x):
        x = self.non_linearity(self.input_layer(x))
        x = self.non_linearity(self.hidden_layer(x))
        return self.output_layer(x)

    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


def load_data(filepath):
    data = pd.read_pickle(filepath)
    data["month"] = data["date"].dt.month + data["date"].dt.year * 100
    return data


def calculate_sharpe_ratio(returns):
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return * np.sqrt(12)


def get_model(model, d, hidden_size):
    if model == "shallow":
        logging.info(f"Using a Shallow Network")
        return ShallowNetwork(d, hidden_size)
    elif model == "deep":
        logging.info(f"Using a Deep Neural Network")
        return DeepNeuralNetwork(d, hidden_size, depth=3)
    elif model == "pilimit":
        logging.info(f"Using a Deep Neural Network with Pi-Limit")
        return PiNetwork(d, 1, r=hidden_size, L=3)
    else:
        raise Exception("Model not Defined")


def run_experiment(hidden_layer, data, feature_cols, window_size, model):
    sharpe_ratios = []
    learning_rates = []
    for lr in np.arange(-18, -9, step=1):

        net = get_model(model, len(feature_cols), hidden_layer)

        optimizer = (
            optim.Adam(net.parameters(), lr=2.0 ** (lr))
            if model != "pilimit"
            else PiSGD(net.parameters(), lr=2.0 ** (lr))
        )
        criterion = MSRR()
        rolling_window = RollingWindow(data, window_size=window_size)
        torch_runner = TorchRunner(
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
        returns = torch_runner.run(net, criterion, optimizer, rolling_window)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        sharpe_ratios.append(sharpe_ratio)
        learning_rates.append(lr)
    return learning_rates, sharpe_ratios


def main(filepath: str, model: str, window_size: int):
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
            size, data, feature_cols, window_size, model
        )

        result_dir = f"results/{model}/{size}"
        os.makedirs(result_dir, exist_ok=True)
        df = pd.DataFrame({"lr": learning_rates, "sharpe": sharpe_ratios})
        df.to_pickle(os.path.join(result_dir, "sharpes.pickle"))

        plt.plot(
            learning_rates, sharpe_ratios, marker="o", label=f"Hidden layers: {size}"
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

    parser.add_argument(
        "--model",
        type=str,
        default="shallow",
        dest="model",
        choices=["shallow", "deep", "pilimit"],
    )
    args = parser.parse_args()

    # 5 years = 60 months
    main(filepath=args.filepath, model=args.model, window_size=5)
