import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mup import MuReadout, set_base_shapes, MuSGD  # Assuming correct import based on your mup library setup

# Define the model class
class MyModel(nn.Module):
    def __init__(self, width, d_out, use_mu_readout=True):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(width, width)  # Standard linear layer with matching input/output width
        if use_mu_readout:
            self.readout = MuReadout(width, d_out)  # Conditional use of MuReadout
        else:
            self.readout = nn.Linear(width, d_out)  # Fallback standard linear layer

    def forward(self, x):
        x = self.layer(x)
        x = self.readout(x)
        return x

# Function to train the model
def train_model(model, optimizer, data_loader, epochs=5):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0
    for _ in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch Loss: {total_loss / len(data_loader)}")
    return total_loss / len(data_loader)

# Parameters for the experiment
widths = [128, 256, 512, 1024, 2048, 4096]
learning_rates = [2**-x for x in range(14, 21)]  # Log2 scale for learning rates

# Adjustments to create a synthetic dataset with varying widths
results = {}
for width in widths:
    inputs = torch.randn(1024, width)
    targets = torch.randn(1024, 10)  # Assuming output dimension is 10
    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Dictionary to store training results for each width
    results[width] = []

    for lr in learning_rates:
        model = MyModel(width, 10, use_mu_readout=True)
        # Ensure base and delta models are initialized with the same width
        base_model = MyModel(width, 10, use_mu_readout=True)
        delta_model = MyModel(width, 10, use_mu_readout=True)
        set_base_shapes(model, base_model, delta=delta_model)

        optimizer = MuSGD(model.parameters(), lr=lr)
        loss = train_model(model, optimizer, data_loader)
        results[width].append(loss)

# Plotting results
plt.figure(figsize=(10, 5))
for width, losses in results.items():
    plt.plot(learning_rates, losses, label=f'Width {width}')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Training Loss')
plt.title('Maximal Update Parametrization')
plt.legend()
plt.xscale('log')
plt.gca().invert_xaxis()  # Invert x-axis to match the given plot style
plt.show()
