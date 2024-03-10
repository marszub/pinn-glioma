from torch import Tensor
import torch
from model.simulationSpace.TimespaceDomain import TimespaceDomain
from model.loss.DiffusionMap import DiffusionMap
import numpy as np


class LoadedDiffusionMap(DiffusionMap):
    def __init__(
        self,
        timespaceDomain: TimespaceDomain,
        loadPath: str,
    ):
        super().__init__(timespaceDomain)
        self.D = torch.from_numpy(np.load(loadPath))

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        self.D = self.D.to(x.device)
        x, y = x.int(), y.int()
        return self.D[x, y]
