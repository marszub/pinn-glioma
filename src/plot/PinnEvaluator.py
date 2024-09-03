from plot.DataProvider import DataProvider
from torch import full_like, tensor
from typing import Callable
from pinn.simulationSpace.UniformSpace import UniformSpace
from typing import Optional
from pinn.Loader import loadModel


class PinnEvaluator(DataProvider):
    def __init__(
        self,
        filename: str,
        space: UniformSpace,
        times: Optional[list] = None
    ):
        super().__init__(space.timespaceDomain)
        self.space = space.to(self.device)

        self.pinn = loadModel(filename).to(self.device)

        if times is None:
            times = [
                self.space.timespaceDomain.timeDomain[0]
                + i
                * (
                    self.space.timespaceDomain.timeDomain[1]
                    - self.space.timespaceDomain.timeDomain[0]
                )
                / self.space.timeResoultion
                for i in range(self.space.timeResoultion)
            ]
            times = tensor(times).to(self.device)
        self.times = times

    def for_each_frame(self, action: Callable):
        x, y, _ = self.space.getInitialPointsKeepDims()
        for time_value in self.times:
            t = full_like(
                x,
                time_value,
            )
            u = self.pinn(x, y, t)
            action(time_value, u)

    def iterator(self):
        import torch
        x_full, y_full, _ = self.space.getInitialPointsKeepDims()
        shape = x_full.size()
        x_full = x_full.reshape(-1, 1)
        y_full = y_full.reshape(-1, 1)
        for time_value in self.times:
            t_full = full_like(
                x_full,
                time_value,
            )
            xs = x_full.split(2**16)
            ys = y_full.split(2**16)
            ts = t_full.split(2**16)
            us = []
            for x, y, t in zip(xs, ys, ts):
                us.append(self.pinn(x, y, t))
            u = torch.cat(us, dim=0).reshape(shape)
            yield time_value, u

    def to(self, device):
        super().to(device)
        self.pinn = self.pinn.to(device)
        self.space = self.space.to(device)
        self.times = self.times.to(device)
