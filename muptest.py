import torch
import torch.nn as nn
from mup import MuReadout, make_base_shapes, set_base_shapes, MuSGD, MuAdam, init

class MyModel(nn.Module):
    def __init__(self, width, d_out):
        super(MyModel, self).__init__()
        self.input_layer = nn.Embedding(num_embeddings=1000, embedding_dim=width)
        self.readout = MuReadout(width, d_out)

    def forward(self, x, query, key):
        x = self.input_layer(x)
        attention_scores = query @ key.T * 8 / x.size(1)  # Using 8/d for backward compatibility
        output = self.readout(x)
        return output, attention_scores

# Model parameters
width = 64
d_out = 10

# Instantiate the base model with minimal width
base_model = MyModel(width=1, d_out=d_out)

# Instantiate a "delta" model with a slightly increased width
delta_model = MyModel(width=2, d_out=d_out)

# Instantiate the target model with the intended training width
model = MyModel(width=width, d_out=d_out)

# Setting base shapes for the model scaling
set_base_shapes(model, base_model, delta=delta_model)

# Optionally save and load shapes to/from a file
# make_base_shapes(base_model, delta_model, 'base_shapes.json')
# set_base_shapes(model, 'base_shapes.json')

# Custom initialization with mup.init
for param in model.parameters():
    init.uniform_(param, -0.1, 0.1)  # Example with uniform initialization

# Setting up the optimizer
optimizer = MuSGD(model.parameters(), lr=0.1)

# Dummy data for demonstration
input_ids = torch.randint(0, 1000, (10,))
query, key = torch.randn(10, width), torch.randn(10, width)

# Training step (simplified)
output, attention_scores = model(input_ids, query, key)
loss = output.mean()  # Simplistic loss calculation for demonstration
loss.backward()
optimizer.step()

print("Output:", output)
print("Attention Scores:", attention_scores)
