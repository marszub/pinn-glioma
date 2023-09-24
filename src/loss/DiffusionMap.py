from torch import Tensor
import torch
from simulationSpace import TimespaceDomain


class DiffusionMap:
    def __init__(
        self, timespaceDomain: TimespaceDomain, device: torch.device
    ):
        self.timespaceDomain = timespaceDomain
        self.device = device

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        res = torch.zeros(x.shape, dtype=x.dtype, device=self.device)
        xDomainBottom, xDomainTop = self.timespaceDomain.spaceDomains[0]
        xDist = (x - xDomainBottom) / (xDomainTop - xDomainBottom) - 0.5
        yDomainBottom, yDomainTop = self.timespaceDomain.spaceDomains[1]
        yDist = (y - yDomainBottom) / (yDomainTop - yDomainBottom) - 0.5
        distSquared = xDist**2 + yDist**2
        res[distSquared < 0.5**2] = 0.13
        res[distSquared < 0.15**2] = 0.013
        return res
