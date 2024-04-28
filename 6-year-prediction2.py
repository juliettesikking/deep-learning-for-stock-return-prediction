import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset

# Assuming MSRR and RollingWindow are defined in MSRR.py and RollingWindow.py respectively
from MSRR import MSRR
from RollingWindow import RollingWindow
from TorchRunner import TorchRunner


class CompleteModel(nn.Module):
    def __init__(self, input_features, output_features):
        super(CompleteModel, self).__init__()
        self.linear = nn.Linear(input_features, output_features)
        self.msrr_loss_module = MSRR(months=1)

    def forward(self, inputs, targets=None):
        outputs = self.linear(inputs)
        if targets is not None:
            loss = self.msrr_loss_module(outputs, targets)
            return outputs, loss
        return outputs


def load_data(file_path):
    data = pd.read_pickle(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values('date', inplace=True)
    return data


def prepare_dataset(data, feature_cols, target_col):
    features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
    targets = torch.tensor(data[target_col].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def main():
    data_path = '/Users/juliette/Documents/bachelor_projet_deep_learning/projet/usa_131_ranked_large_mega.pickle'  # Update with your actual data file path
    data = load_data(data_path)

    feature_cols = data.columns.difference(['id', 'date', 'size_grp', 'r_1'])
    target_col = 'r_1'

    model = CompleteModel(input_features=len(feature_cols), output_features=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_date = data['date'].min()
    end_date = data['date'].max()
    predictions_list = []

    while start_date + timedelta(days=730) <= end_date:
        train_end = start_date + timedelta(days=730)
        test_start = train_end + timedelta(days=1)

        if test_start + timedelta(days=30) > end_date:
            break

        train_data = data[(data['date'] >= start_date) & (data['date'] <= train_end)]
        test_data = data[(data['date'] >= test_start) & (data['date'] < test_start + timedelta(days=30))]

        train_loader = prepare_dataset(train_data, feature_cols, target_col)
        test_loader = prepare_dataset(test_data, feature_cols, target_col)

        for epoch in range(10):  # number of epochs
            model.train()
            for features, targets in train_loader:
                optimizer.zero_grad()
                outputs, loss = model(features, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for features, _ in test_loader:
                predictions = model(features)
                # Store predictions with corresponding date
                for i, prediction in enumerate(predictions):
                    predictions_list.append({
                        'date': (test_start + timedelta(days=i)).strftime("%Y-%m"),
                        'prediction': prediction.item()
                    })

        # Move to next window
        start_date += timedelta(days=730)

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions_list)

    # Output predictions
    print(predictions_df)

    # Optionally, save predictions to a file
    predictions_df.to_csv('predictions_with_dates.csv', index=False)


if __name__ == "__main__":
    main()
