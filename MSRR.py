import torch
import torch.nn as nn


class MSRR(nn.Module):
    def __init__(self, months=1):
        super(MSRR, self).__init__()
        self.months = months

    def forward(self, inputs, targets):
        # b = (1 - F^\top R_{t+1}) \in \R^{N_t}
        loss = (1 - torch.matmul(inputs.T, targets)) ** 2 / (
            self.months * targets.shape[0]
        )

        return loss
