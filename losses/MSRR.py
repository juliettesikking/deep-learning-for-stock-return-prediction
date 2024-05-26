import torch
import torch.nn as nn


class MSRR(nn.Module):
    def __init__(self, months=1):
        super(MSRR, self).__init__()
        self.months = months

    def forward(self, inputs, targets):
        # Inputs is a vector
        # targets is a vector

        # Thus, 1 - (inputs'targets)**2 is a scalar
        loss = (
            1
            - torch.matmul(
                inputs.T,
                targets,
            )
        ) ** 2

        return loss
