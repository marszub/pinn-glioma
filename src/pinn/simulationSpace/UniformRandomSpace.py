import torch
from pinn.simulationSpace.RandomSpace import RandomSpace


class UniformRandomSpace(RandomSpace):
    def getT(self, n: int) -> torch.Tensor:
        tMin, tMax = self.timespaceDomain.timeDomain
        return self.__getVector(n, tMin, tMax)
