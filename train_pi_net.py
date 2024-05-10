import numpy as np
import time

torch.manual_seed(3133)
np.random.seed(3331)
device = "cuda" if torch.cuda.is_available() else "cpu"

data = torch.linspace(-np.pi, np.pi, 100, device=device).reshape(-1, 1)
labels = torch.sin(data) #.reshape(-1)
data = torch.cat([data, torch.ones_like(data, device=device)], dim=1)

d_in = 2
d_out = 3
r = 20
L = 1
bias_alpha = .5
batch_size = 50
net = InfMLP(d_in, d_out, r, L, device=device, bias_alpha=bias_alpha )

from pilimit_lib.inf.optim import PiSGD, store_pi_grad_norm_, clip_grad_norm_
import sys

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
    # stage_grad(net)

    if epoch % accum_steps == 0:
        # unstage_grad(net)

        if gclip:
            store_pi_grad_norm_(net.modules())
            clip_grad_norm_(net.parameters(), gclip)

        optimizer.step()

    # print("Memory used", torch.cuda.memory_reserved() / 1e9, torch.cuda.max_memory_reserved()  / 1e9)
    print("Network A size", net.layers[1].A.shape[0])
print("time", time.time() - tic)