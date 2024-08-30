import torch
from typing import Tuple

from model.TimespaceDomain import TimespaceDomain
from pinn.sample.SampleSpace import SampleSpace


class UniformRandom(SampleSpace):
    def __init__(self, timespace_domain: TimespaceDomain, sample_size: int):
        super().__init__(timespace_domain)
        self.sample_size = sample_size

    def get_points(self):
        x = self.__rand_uniform(self.timespace_domain.spaceDomains[0])
        y = self.__rand_uniform(self.timespace_domain.spaceDomains[1])
        t = self.__rand_uniform(self.timespace_domain.timeDomain)
        return x, y, t

    def __rand_uniform(self, domain: Tuple[float, float]):
        min_val, max_val = domain
        vec = torch.rand(self.sample_size, 1, device=self.device)
        vec = (max_val - min_val) * vec + min_val
        vec.requires_grad = True
        return vec
