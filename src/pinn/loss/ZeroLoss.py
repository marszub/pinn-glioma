import torch

from pinn.Pinn import PINN


class ZeroLoss:
    def __call__(self, pinn: PINN):
        return torch.tensor(0.0, device=pinn.device())
