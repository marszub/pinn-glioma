import torch
from pinn.simulationSpace.RandomSpace import RandomSpace


class ExponentialRandomSpace(RandomSpace):
    def getT(self, n: int):
        tMin, tMax = self.timespaceDomain.timeDomain
        rate = torch.tensor(0.5).to(self.device)
        distribution = torch.distributions.exponential.Exponential(rate)
        t = tMin + torch.fmod(distribution.sample((n, 1)), tMax - tMin)
        t.requires_grad = True
        return t
