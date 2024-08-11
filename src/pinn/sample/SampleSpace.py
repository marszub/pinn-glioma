import torch

from model.TimespaceDomain import TimespaceDomain


class SampleSpace:
    def __init__(self, timespace_domain: TimespaceDomain):
        self.device = torch.device("cpu")
        self.timespace_domain = timespace_domain

    def get_points(self):
        """
        Generate n points from within timespace domain.
        Returns x, y, t tensors of size (n, 1) each.
        """
        raise NotImplementedError("Abstract method")

    def to(self, device):
        self.device = device
        return self
