form plot.DataProvider import DataProvider
from torch import load
from pinn.Configuration import Configuration


class PinnEvaluator(DataProvider):
    def __init__(self, filename: str, space: UniformSpace, config: Configuration):
        self.super(space.timespaceDomain)
        self.space = space

        self.pinn = config.getNeuralNetwork()
        self.pinn.load_state_dict(load(filename))
        self.pinn.eval()

    def for_each_frame(self, action: Callable):
        x, y, _ = self.space.getInitialPointsKeepDims()
        for i in range(self.space.timeResoultion):
            time_value = (
                self.space.timespaceDomain.timeDomain[0]
                + i
                * (
                    self.space.timespaceDomain.timeDomain[1]
                    - self.space.timespaceDomain.timeDomain[0]
                )
                / self.space.timeResoultion
            )
            t = full_like(
                x,
                time_value,
            )
            u = self.pinn(x, y, t)
            action(t, u)
