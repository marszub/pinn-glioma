from torch import Tensor
import torch
from model.simulationSpace.TimespaceDomain import TimespaceDomain
from model.loss.DiffusionMap import DiffusionMap
import numpy as np


class LoadedDiffusionMap(DiffusionMap):
    def __init__(
        self,
        timespaceDomain: TimespaceDomain,
        device: torch.device,
        loadPath: str,
    ):
        super().__init__(timespaceDomain, device)
        self.D = torch.from_numpy(np.load(loadPath)).to(self.device)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = x.int(), y.int()
        return self.D[x, y]
