import torch

from pinn.Pinn import PINN


class ZeroLoss:
    def __init__(self):
        self.device = torch.device("cpu")

    def __call__(self, pinn: PINN):
        return torch.tensor(0.0, device=self.device)

    def to(self, device):
        self.device = device
