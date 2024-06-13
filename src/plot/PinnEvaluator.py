from plot.DataProvider import DataProvider
from torch import load, full_like, tensor
from pinn.PinnConfig import PinnConfig
from typing import Callable
from pinn.simulationSpace.UniformSpace import UniformSpace
from typing import Optional


class PinnEvaluator(DataProvider):
    def __init__(
        self,
        filename: str,
        space: UniformSpace,
        config: PinnConfig,
        times: Optional[list] = None
    ):
        super().__init__(space.timespaceDomain)
        self.space = space

        self.pinn = config.getNeuralNetwork()
        self.pinn.load_state_dict(load(filename))
        self.pinn.eval()

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
        x, y, _ = self.space.getInitialPointsKeepDims()
        for time_value in self.times:
            t = full_like(
                x,
                time_value,
            )
            u = self.pinn(x, y, t)
            yield time_value, u

    def to(self, device):
        super().to(device)
        self.pinn = self.pinn.to(device)
        self.space = self.space.to(device)
        self.times = self.times.to(device)
