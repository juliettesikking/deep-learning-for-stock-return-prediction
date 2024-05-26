import torch
from torch import nn
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from experiments.networks.networks import PiNet


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

        self.register_param_buffer("first_layer_alpha", first_layer_alpha)
        self.register_param_buffer("last_layer_alpha", last_layer_alpha)
        self.register_param_buffer("bias_alpha", bias_alpha)
        if last_bias_alpha is None:
            last_bias_alpha = bias_alpha
        self.register_param_buffer("last_bias_alpha", last_bias_alpha)
        self.layernorm = layernorm

        self.layers = nn.ModuleList()

        self.layers.append(
            InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device)
        )
        for n in range(1, L + 1):
            self.layers.append(
                InfPiLinearReLU(
                    r,
                    device=device,
                    bias_alpha=bias_alpha,
                    layernorm=layernorm,
                    cuda_batch_size=cuda_batch_size,
                )
            )

        self.layers.append(
            InfPiLinearReLU(
                r,
                r_out=d_out,
                output_layer=True,
                bias_alpha=None,
                device=device,
                layernorm=layernorm,
                cuda_batch_size=cuda_batch_size,
            )
        )

    def register_param_buffer(self, param_name, value):
        # Register individual floats as buffers for later saving/loading
        self.register_buffer(
            param_name, torch.tensor(value, dtype=torch.get_default_dtype())
        )

    def forward(self, x):
        for n in range(0, self.L + 2):
            x = self.layers[n](x)
            if n == 0:
                x *= self.first_layer_alpha
            if n == self.L + 1:
                x *= self.last_layer_alpha
        return x
