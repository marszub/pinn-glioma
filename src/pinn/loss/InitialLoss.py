from model.TimespaceDomain import TimespaceDomain
from pinn.loss.Function import f
from pinn.Pinn import PINN
import torch
from typing import Tuple, Callable


class InitialLoss:
    def __init__(
        self,
        timespace_domain: TimespaceDomain,
        sample_size: int,
        initial_condition: Callable,
    ):
        self.device = torch.device("cpu")
        self.timespace_domain = timespace_domain
        self.sample_size = sample_size
        self.initial_condition = initial_condition

    def __call__(self, pinn: PINN):
        x_domain = self.timespace_domain.spaceDomains[0]
        y_domain = self.timespace_domain.spaceDomains[1]
        t_min = self.timespace_domain.timeDomain[0]
        x = self.sample(x_domain)
        y = self.sample(y_domain)
        t0 = torch.full((self.sample_size, 1), t_min,
                        requires_grad=True, device=self.device)
        pinn_init = self.initial_condition(x, y)
        loss = f(pinn, x, y, t0) - pinn_init
        return loss.pow(2).mean()

    def sample(self, domain: Tuple[float, float]):
        minVal, maxVal = domain
        vec = torch.rand(self.sample_size, 1, device=self.device)
        vec = (maxVal - minVal) * vec + minVal
        vec.requires_grad = True
        return vec

    def to(self, device):
        self.device = device
