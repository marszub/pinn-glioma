import torch
from typing import Callable


class DataProvider:
    def __init__(self, timespace_domain):
        self.timespace_domain = timespace_domain

    def for_each_frame(self, action: Callable):
        raise NotImplementedError("Abstract method")

    def iterator(self):
        raise NotImplementedError("Abstract method")

    def get_size_over_time(self):
        times = []
        sums = []
        for t, u in self.iterator():
            points_x = u.shape[0]
            points_y = u.shape[1]
            times.append(t)
            sums.append(torch.sum(u))

        xSpaceSize = (
            self.timespace_domain.spaceDomains[0][1]
            - self.timespace_domain.spaceDomains[0][0]
        )
        xpointsPerLengthUnit = points_x / xSpaceSize
        ySpaceSize = (
            self.timespace_domain.spaceDomains[1][1]
            - self.timespace_domain.spaceDomains[1][0]
        )
        ypointsPerLengthUnit = points_y / ySpaceSize
        points_per_space_unit = ypointsPerLengthUnit * xpointsPerLengthUnit

        sizes = torch.tensor(sums) / points_per_space_unit
        return torch.tensor(times), sizes
