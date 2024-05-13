import torch
from pinn.simulationSpace.SampleSpace import SampleSpace

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model.TimespaceDomain import TimespaceDomain

class RandomSpace(SampleSpace):
    def __init__(self, timespaceDomain: "TimespaceDomain", initialSize: int, interiorSize: int, boundarySize: int):
        super().__init__(timespaceDomain)
        self.initialSize = initialSize
        self.interiorSize = interiorSize
        self.boundarySize = boundarySize

    def getInitialPoints(self):
        """Generates tensor of points convering initial condition"""
        x = self.__getX(self.initialSize)
        y = self.__getY(self.initialSize)
        tMin, _ = self.timespaceDomain.timeDomain
        t0 = torch.full_like(x, tMin, requires_grad=True, device=self.device)
        return x, y, t0

    def getInteriorPoints(self):
        """Generates tensor of points convering interior of simulation"""
        return self.__getX(self.interiorSize), self.__getY(self.interiorSize), self.__getT(self.interiorSize)

    def getBoundaryPoints(self):
        """Generates tensor of points convering space boundary"""
        x = self.__getX(self.boundarySize)
        y = self.__getY(self.boundarySize)
        t = self.__getT(self.boundarySize)

        xMin, xMax = self.timespaceDomain.spaceDomains[0]
        x0 = torch.full_like(x, xMin, requires_grad=True, device=self.device)
        x1 = torch.full_like(x, xMax, requires_grad=True, device=self.device)

        yMin, yMax = self.timespaceDomain.spaceDomains[1]
        y0 = torch.full_like(y, yMin, requires_grad=True, device=self.device)
        y1 = torch.full_like(y, yMax, requires_grad=True, device=self.device)

        down = (x, y0, t)
        up = (x, y1, t)
        left = (x0, y, t)
        right = (x1, y, t)

        return down, up, left, right

    def __getVector(self, n: int, minVal: float, maxVal: float):
        vec = torch.rand(n, 1, device=self.device)
        vec = (maxVal - minVal) * vec + minVal
        vec.requires_grad = True
        return vec
    
    def __getX(self, n: int):
        xMin, xMax = self.timespaceDomain.spaceDomains[0]
        return self.__getVector(n, xMin, xMax)
    
    def __getY(self, n: int):
        yMin, yMax = self.timespaceDomain.spaceDomains[1]
        return self.__getVector(n, yMin, yMax)
    
    def __getT(self, n: int):
        tMin, tMax = self.timespaceDomain.timeDomain
        return self.__getVector(n, tMin, tMax)
