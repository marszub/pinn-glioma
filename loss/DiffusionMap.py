from torch import Tensor
import torch
from simulationSpace import TimespaceDomain


class DiffusionMap:
    def __init__(self, timespaceDomain: TimespaceDomain, device: torch.device):
        self.timespaceDomain = timespaceDomain
        self.device = device

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        res = torch.zeros(x.shape, dtype=x.dtype, device=self.device)
        dist = ((x-self.timespaceDomain.spaceDomains[0][0])/(self.timespaceDomain.spaceDomains[0][1] - self.timespaceDomain.spaceDomains[0][0]) - 0.5) ** 2 + ((y-self.timespaceDomain.spaceDomains[1][0])/(self.timespaceDomain.spaceDomains[1][1] - self.timespaceDomain.spaceDomains[1][0]) - 0.5) ** 2
        res[dist < 0.25] = 0.13
        res[dist < 0.02] = 0.013
        return res