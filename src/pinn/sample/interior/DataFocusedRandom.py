import torch
from typing import Tuple

from model.TimespaceDomain import TimespaceDomain
from pinn.sample.SampleSpace import SampleSpace


class DataFocusedRandom(SampleSpace):
    def __init__(
        self,
        timespace_domain: TimespaceDomain,
        sample_size: int,
        times: list,
        rate: float = 1.0,
    ):
        super().__init__(timespace_domain)
        self.sample_size = sample_size
        self.times = torch.tesnor(times, device=self.device)
        self.times = self.times.reshape((-1,))
        self.rate = torch.tesnor(rate, device=self.device)
        self.mean = torch.tensor(0.0, device=self.device)

    def get_points(self):
        x = self.__rand_uniform(self.timespace_domain.spaceDomains[0])
        y = self.__rand_uniform(self.timespace_domain.spaceDomains[1])
        time_idx = torch.randint(
            0, self.times.shape[0], (self.sample_size, 1), device=self.device)
        tMin, tMax = self.timespaceDomain.timeDomain
        support_size = tMax - tMin
        sample = torch.distributions.laplace.Laplace(
            self.mean, self.rate).sample((self.sample_size, 1))
        t = tMin + torch.fmod(
            self.times[time_idx] + sample*support_size, support_size
        )
        t.requires_grad = True
        return x, y, t

    def __rand_uniform(self, domain: Tuple[float, float]):
        min_val, max_val = domain
        vec = torch.rand(self.sample_size, 1, device=self.device)
        vec = (max_val - min_val) * vec + min_val
        vec.requires_grad = True
        return vec

    def to(self, device):
        self.rate = self.rate.to(device)
        self.mean = self.mean.to(device)
        self.times = self.times.to(device)
        return super().to(device)
