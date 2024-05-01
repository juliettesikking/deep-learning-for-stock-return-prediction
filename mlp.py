import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner
import numpy as np

# Assuming a CompleteModel class exists
class CompleteModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers: list):
        super(CompleteModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_layers[0])
        self.hidden_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_layers[i + 1], hidden_layers[i + 2], bias=True)
                for i in range(len(hidden_layers) - 2)
            ]
        )
        self.output_layer = torch.nn.Linear(hidden_layers[-1], 1, bias=False)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x):
        x = self.non_linearity(self.linear(x))
        for hidden_layer in self.hidden_layers:
            x = self.non_linearity(hidden_layer(x))
        # without non-linear activation function
        output = self.output_layer(x)
        return output


def load_data(filepath):
    # Load your data here
    data = pd.read_pickle(filepath)
    data['month'] = data['date'].dt.month + data['date'].dt.year * 100
    return data


def prepare_dataset(data, feature_cols, target_col):
    features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
    targets = torch.tensor(data[target_col].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def main(filepath: str, output_dir: str):
    data = load_data(filepath)

    feature_cols = [col for col in data.columns if col not in ['id', 'date', 'size_grp', 'r_1', 'month']]
    target_col = 'r_1'
    for lr in np.arange(-18, -9, step=1):
        model = CompleteModel(input_dim=len(feature_cols), hidden_layers=[16, 16, 16])

        optimizer = optim.Adam(model.parameters(), lr=2**(-lr))
        criterion = MSRR()

        rolling_window = RollingWindow(data, window_size=12)  # 24 months window size
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

        # Example of how to train and predict - add your loop here
        # Use torch_runner.run() as per your TorchRunner's implementation
        torch_runner.run(model, criterion, optimizer, rolling_window)


if __name__ == "__main__":
    torch.manual_seed(1234)
    # Reads from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", dest="filepath", type=str, default='/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle')
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="./")
    
    args = parser.parse_args()
    main(filepath=args.filepath, output_dir=args.output_dir)
