import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner


# Assuming a CompleteModel class exists
class CompleteModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CompleteModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.linear(x)
        norms = torch.norm(self.linear.weight, dim=1)  # Example to compute norms
        return output, norms


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


def main():
    filepath = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'
    data = load_data(filepath)

    feature_cols = [col for col in data.columns if col not in ['id', 'date', 'size_grp', 'r_1', 'month']]
    target_col = 'r_1'

    model = CompleteModel(input_dim=len(feature_cols), output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = MSRR()

    rolling_window = RollingWindow(data, window_size=24)  # 24 months window size
    torch_runner = TorchRunner(
        rolling_window=True,
        window_size=24,
        end_year=2023,
        output='/path/to/output',
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
    main()
