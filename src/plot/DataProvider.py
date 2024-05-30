import torch
from typing import Callable


class DataProvider:
    def __init__(self, timespace_domain):
        self.timespace_domain = timespace_domain

    def for_each_frame(self, action: Callable):
        raise NotImplementedError("Abstract method")

    def get_size_over_time(self):
        class Iteration:
            def __init__(self):
                self.times = []
                self.sums = []

            def __call__(self, t, u):
                self.points_x = u.shape[0]
                self.points_y = u.shape[1]
                self.times.append(t)
                self.sums.append(torch.sum(u))

        iteration = Iteration()
        self.for_each_frame(iteration)
        xSpaceSize = (
            self.timespace_domain.spaceDomains[0][1]
            - self.timespace_domain.spaceDomains[0][0]
        )
        xpointsPerLengthUnit = iteration.points_x / xSpaceSize
        ySpaceSize = (
            self.timespace_domain.spaceDomains[1][1]
            - self.timespace_domain.spaceDomains[1][0]
        )
        ypointsPerLengthUnit = iteration.points_y / ySpaceSize
        points_per_space_unit = ypointsPerLengthUnit * xpointsPerLengthUnit

        sizes = torch.tensor(iteration.sums) / points_per_space_unit
        return torch.tensor(iteration.times), sizes
