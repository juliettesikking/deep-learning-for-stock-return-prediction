import torch
import torch.nn as nn


class MSRR(nn.Module):
    def __init__(self, months=1):
        super(MSRR, self).__init__()
        self.months = months

    def forward(self, inputs, targets):
        inputs = inputs.view(1, -1)  # Convert to (1, N) row vector
        targets = targets.view(-1, 1)  # Convert to (N, 1) column vector

        # Perform matrix multiplication
        loss = (1 - torch.matmul(inputs, targets)) ** 2

        return loss
