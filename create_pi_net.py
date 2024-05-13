import torch
from torch import nn
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from pilim.experiments.networks.networks import PiNet
import numpy as np
import time

class InfMLP(PiNet):
    def __init__(self, d_in, d_out, r, L, first_layer_alpha=1, last_layer_alpha=1, bias_alpha=1, last_bias_alpha=None, layernorm=False, cuda_batch_size=None, device="cpu"):
        super(InfMLP, self).__init__()
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
        self.layers.append(InfPiInputLinearReLU(d_in, r, bias_alpha=bias_alpha, device=device))
        for n in range(1, L + 1):
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm, cuda_batch_size=cuda_batch_size))
        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device, layernorm=layernorm, cuda_batch_size=cuda_batch_size))

    def register_param_buffer(self, param_name, value):
        self.register_buffer(param_name, torch.tensor(value, dtype=torch.get_default_dtype()))

    def forward(self, x):
        for n in range(0, self.L + 2):
            x = self.layers[n](x)
            if n == 0:
                x *= self.first_layer_alpha
            if n == self.L + 1:
                x *= self.last_layer_alpha
        return x

if __name__ == "__main__":
    torch.manual_seed(3133)
    np.random.seed(3331)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.linspace(-np.pi, np.pi, 100, device=device).reshape(-1, 1)
    labels = torch.sin(data)
    data = torch.cat([data, torch.ones_like(data, device=device)], dim=1)

    d_in = 2
    d_out = 3
    r = 20
    L = 1
    bias_alpha = .5

    net = InfMLP(d_in, d_out, r, L, device=device, bias_alpha=bias_alpha)

    from pilimit_lib.inf.optim import PiSGD, store_pi_grad_norm_, clip_grad_norm_

    net.train()
    epoch = 20
    accum_steps = 1
    gclip = .1
    optimizer = PiSGD(net.parameters(), lr=.02)
    tic = time.time()
    for epoch in range(epoch):
        if epoch % accum_steps == 0:
            optimizer.zero_grad()
            net.zero_grad()

        prediction = net(data)

        loss = torch.sum((prediction - labels) ** 2) ** .5

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

        loss.backward()

        if epoch % accum_steps == 0:
            if gclip:
                store_pi_grad_norm_(net.modules())
                clip_grad_norm_(net.parameters(), gclip)

            optimizer.step()

        print("Network A size", net.layers[1].A.shape[0])
    print("time", time.time() - tic)
