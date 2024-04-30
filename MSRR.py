import torch
import torch.nn as nn

class MSRR(nn.Module):
    def __init__(self, months=1):
        super(MSRR, self).__init__()
        self.months = months

    def forward(self, inputs, targets):
        # Ensure inputs are [batch_size, num_features] and targets are [batch_size, 1]
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))  # Flatten if needed
        if targets.dim() > 2:
            targets = targets.view(-1, 1)  # Reshape targets correctly

        # Use matmul for dot product and handle the transposition with .mT if inputs is not 2-dimensional
        loss = (1 - torch.matmul(inputs.mT if inputs.dim() == 2 else inputs.permute(*torch.arange(inputs.ndim - 1, -1, -1)), targets)) ** 2
        loss = loss.sum() / (self.months * targets.shape[0])  # Normalize loss

        return loss
