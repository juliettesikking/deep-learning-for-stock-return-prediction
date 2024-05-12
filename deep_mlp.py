import argparse
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from losses.MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner


class DeepNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.layer3 = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.output_layer = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x):
        x = self.non_linearity(self.layer1(x))
        x = self.non_linearity(self.layer2(x))
        x = self.non_linearity(self.layer3(x))
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


def run_experiment(hidden_dim, data, feature_cols, window_size):
    sharpe_ratios = []
    learning_rates = []
    for lr in np.arange(-18, -9, step=1):
        model = DeepNetwork(
            input_dim=len(feature_cols),
            hidden_dim=hidden_dim,
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
