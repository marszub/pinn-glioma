import torch

from model.TimespaceDomain import TimespaceDomain
from pinn.sample.SampleSpace import SampleSpace


class UniformDeterministic(SampleSpace):
    def __init__(
        self,
        timespace_domain: TimespaceDomain,
        space_resolution: int,
        time_resolution: int,
    ):
        super().__init__(timespace_domain)
        self.space_resolution = space_resolution
        self.time_resolution = time_resolution

    def get_points(self):
        x_domain = self.timespaceDomain.spaceDomains[0]
        y_domain = self.timespaceDomain.spaceDomains[1]
        t_domain = self.timespaceDomain.timeDomain
        x_raw = torch.linspace(
            x_domain[0],
            x_domain[1],
            steps=self.spaceResoultion,
            requires_grad=True,
        )
        y_raw = torch.linspace(
            y_domain[0],
            y_domain[1],
            steps=self.spaceResoultion,
            requires_grad=True,
        )
        t_raw = torch.linspace(
            t_domain[0],
            t_domain[1],
            steps=self.timeResoultion,
            requires_grad=True,
        )
        grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

        x = grids[0].reshape(-1, 1).to(self.device)
        y = grids[1].reshape(-1, 1).to(self.device)
        t = grids[2].reshape(-1, 1).to(self.device)
        return x, y, t
