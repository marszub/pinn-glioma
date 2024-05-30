import torch
from pinn.simulationSpace.RandomSpace import RandomSpace


class ExponentialRandomSpace(RandomSpace):
    def __getT(self, n: int):
        tMin, tMax = self.timespaceDomain.timeDomain
        distribution = torch.distributions.exponential.Exponential(0.5)
        return tMin + torch.fmod(distribution.sample_n(n), tMax - tMin)
