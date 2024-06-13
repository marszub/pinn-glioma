import torch
from typing import Callable


class DataProvider:
    def __init__(self, timespace_domain):
        self.timespace_domain = timespace_domain
        self.device = torch.device("cpu")

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

        sizes = (
            torch.tensor(sums) /
            self.timespace_domain.get_points_per_space_unit(
                points_x * points_y
            )
        )

        return torch.tensor(times), sizes

    def to(self, device):
        self.device = device
        return self
