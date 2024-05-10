import torch
from torch import nn
from pilimit_lib.inf.layers import InfPiInputLinearReLU, InfPiLinearReLU
from experiments.networks.networks import PiNet
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import numpy as np
import time

class InfMLP(PiNet):
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
            device="cpu"):
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
            self.layers.append(InfPiLinearReLU(r, device=device, bias_alpha=bias_alpha, layernorm=layernorm,
                                               cuda_batch_size=cuda_batch_size))

        self.layers.append(InfPiLinearReLU(r, r_out=d_out, output_layer=True, bias_alpha=last_bias_alpha, device=device,
                                           layernorm=layernorm, cuda_batch_size=cuda_batch_size))

    def register_param_buffer(self, param_name, value):
        # Register individual floats as buffers for later saving/loading
        self.register_buffer(param_name, torch.tensor(value, dtype=torch.get_default_dtype()))

    def forward(self, x):
        for n in range(0, self.L + 2):
            x = self.layers[n](x)
            if n == 0:
                x *= self.first_layer_alpha
            if n == self.L + 1:
                x *= self.last_layer_alpha
        return x

total_samples = 10
batch_size = 1

transform_list = []
transform_list.extend([transforms.ToTensor()])

transform_list.extend([transforms.Normalize([0.49137255, 0.48235294, 0.44666667], [0.24705882, 0.24352941, 0.26156863])])
transform = transforms.Compose(transform_list)

trainset = datasets.CIFAR10(root=".", train=True,
                                        download=True, transform=transform)

np.random.seed(0) # reproducability of subset
indices = np.random.choice(range(50000), size=total_samples, replace=False).tolist()
trainset = data_utils.Subset(trainset, indices)
print("Using subset of", len(trainset), "training samples")
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

testset = datasets.CIFAR10(root=".", train=False,
                                      download=True, transform=transform)
np.random.seed(0) # reproducability of subset
indices = np.random.choice(range(50000), size=total_samples, replace=False).tolist()
testset = data_utils.Subset(testset, indices)
print("Using subset of", len(testset), "testing samples")
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)
