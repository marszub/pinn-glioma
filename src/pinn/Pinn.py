import torch
from torch import nn


class PINN(nn.Module):
    """
    Simple neural network accepting two features as input and returning a single output
    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """

    def __init__(self, layers: int, neuronsPerLayer: int, act=nn.Tanh()):
        super().__init__()
        self.layer_in = nn.Linear(3, neuronsPerLayer)
        self.layer_out = nn.Linear(neuronsPerLayer, 1)
        num_middle = layers - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(neuronsPerLayer, neuronsPerLayer)
             for _ in range(num_middle)]
        )
        self.act = act

    def forward(self, x, y, t):
        x_stack = torch.cat([x, y, t], dim=-1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)
        return logits

    def device(self):
        return next(self.parameters()).device
