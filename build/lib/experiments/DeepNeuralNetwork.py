import torch


class DeepNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, depth: int):
        super(DeepNeuralNetwork, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_size, bias=True)

        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size, bias=True) for i in range(depth)]
        )
        self.output_layer = torch.nn.Linear(hidden_size, 1, bias=False)
        self.non_linearity = torch.nn.ReLU()

    def forward(self, x):
        x = self.non_linearity(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.non_linearity(hidden_layer(x))
        return self.output_layer(x)

    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
