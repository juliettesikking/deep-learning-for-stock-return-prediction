import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging

from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from pilimit_lib.inf.optim import PiSGD

logging.basicConfig(level=logging.INFO)


class PiNet(nn.Module):
    def __init__(self):
        super(PiNet, self).__init__()


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

        # Corrected buffer registration
        self.register_buffer("first_layer_alpha", torch.tensor(first_layer_alpha, dtype=torch.float))
        self.register_buffer("last_layer_alpha", torch.tensor(last_layer_alpha, dtype=torch.float))
        self.register_buffer("bias_alpha", torch.tensor(bias_alpha, dtype=torch.float))
        last_bias_alpha = last_bias_alpha if last_bias_alpha is not None else bias_alpha
        self.register_buffer("last_bias_alpha", torch.tensor(last_bias_alpha, dtype=torch.float))

        self.layernorm = layernorm

        self.layers = nn.ModuleList()
        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=self.bias_alpha, device=device))
        for n in range(1, L):
            self.layers.append(
                InfPiLinearReLU(r, r, bias_alpha=self.bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size,
                                device=device)
            )
        self.layers.append(nn.Linear(r, d_out, bias=False))

    def forward(self, x):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if n == 0:
                x *= self.first_layer_alpha
            elif n == len(self.layers) - 1:
                x *= self.last_layer_alpha
        return x


def run_experiment(hidden_size):
    learning_rates = np.power(2.0, np.arange(-18, -9))
    losses = []

    model = PiNetwork(784, 10, r=hidden_size, L=3)  # Adjust for your dataset/features
    criterion = torch.nn.CrossEntropyLoss()

    for lr in learning_rates:
        optimizer = PiSGD(model.parameters(), lr=lr)

        total_loss = 0
        for _ in range(100):  # Dummy iteration count for batches
            inputs = torch.randn(32, 784)  # Dummy input for 32 samples, 784 features
            targets = torch.randint(0, 10, (32,))  # Dummy targets for 10 classes
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / 100
        losses.append(avg_loss)
        print(f"Learning Rate: {lr:.6f}, Average Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(np.log2(learning_rates), losses, marker='o', label='Loss vs. Learning Rate')
    plt.xlabel("Log2 Learning Rate")
    plt.ylabel("Average Loss")
    plt.title("Learning Rate vs Loss for Pi-Limit Model")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    hidden_size = 128  # Example hidden layer size
    run_experiment(hidden_size)


if __name__ == "__main__":
    main()
