import torch
from typing import NamedTuple
from pinn.simulationSpace.SampleSpace import SampleSpace

from model.TimespaceDomain import TimespaceDomain


class SampleSizes(NamedTuple):
    initial: int
    interior: int
    boundary: int


class RandomSpace(SampleSpace):
    def __init__(self, timespaceDomain: TimespaceDomain):
        super().__init__(timespaceDomain)
        self.initialSize = sizes.initial
        self.interiorSize = sizes.interiorSize
        self.boundarySize = sizes.boundarySize

    def getInteriorPoints(self, n: int):
        """Generates tensor of points convering interior of simulation"""
        xMin, xMax = self.timespaceDomain.spaceDomains[0]
        x = self.getUniformVector(n, xMin, xMax)
        yMin, yMax = self.timespaceDomain.spaceDomains[1]
        y = self.getUniformVector(n, yMin, yMax)
        t = self.get_interior_t(n)
        return x, y, t

    def getUniformVector(self, n: int, minVal: float, maxVal: float):
        vec = torch.rand(n, 1, device=self.device)
        vec = (maxVal - minVal) * vec + minVal
        vec.requires_grad = True
        return vec

    def get_interior_t(self, n: int) -> torch.Tensor:
        raise NotImplementedError("Use of abstract method")
