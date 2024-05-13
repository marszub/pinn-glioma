import torch

from model.TimespaceDomain import TimespaceDomain

class SampleSpace:
    def __init__(self, timespaceDomain: TimespaceDomain):
        self.device = torch.device("cpu")
        self.timespaceDomain = timespaceDomain

    def getInitialPoints(self):
        """Generates tensor of points convering initial condition"""
        pass

    def getInteriorPoints(self):
        """Generates tensor of points convering interior of simulation"""
        pass

    def getBoundaryPoints(self):
        """Generates tensor of points convering space boundary"""
        pass
    
    def to(self, device):
        self.device = device
