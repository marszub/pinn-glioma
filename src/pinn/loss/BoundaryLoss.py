from pinn.loss.Function import f
import torch
from pinn.Pinn import PINN
from typing import Tuple
from model.TimespaceDomain import TimespaceDomain


class BoundaryLoss:
    def __init__(self, timespace_domain: TimespaceDomain, sample_size: int):
        self.device = torch.device("cpu")
        self.timespace_domain = timespace_domain
        self.sample_size = sample_size

    def __call__(self, pinn: PINN):
        x_domain = self.timespace_domain.spaceDomains[0]
        x0 = torch.full(
            (self.sample_size, 1),
            x_domain[0],
            requires_grad=True,
            device=self.device,
        )
        x1 = torch.full(
            (self.sample_size, 1),
            x_domain[1],
            requires_grad=True,
            device=self.device,
        )

        y_domain = self.timespace_domain.spaceDomains[1]
        y0 = torch.full(
            (self.sample_size, 1),
            y_domain[0],
            requires_grad=True,
            device=self.device,
        )
        y1 = torch.full(
            (self.sample_size, 1),
            y_domain[1],
            requires_grad=True,
            device=self.device,
        )

        t_domain = self.timespace_domain.timeDomain

        loss_down = f(pinn, self.sample(x_domain), y0, self.sample(t_domain))
        loss_up = f(pinn, self.sample(x_domain), y1, self.sample(t_domain))
        loss_left = f(pinn, x0, self.sample(y_domain), self.sample(t_domain))
        loss_right = f(pinn, x1, self.sample(y_domain), self.sample(t_domain))

        return (
            loss_down.pow(2).mean()
            + loss_up.pow(2).mean()
            + loss_left.pow(2).mean()
            + loss_right.pow(2).mean()
        )

    def sample(self, domain: Tuple[float, float]):
        minVal, maxVal = domain
        vec = torch.rand(self.sample_size, 1, device=self.device)
        vec = (maxVal - minVal) * vec + minVal
        vec.requires_grad = True
        return vec

    def to(self, device):
        self.device = device
