from torch import Tensor
import torch
from model.TimespaceDomain import TimespaceDomain
from model.DiffusionMap import DiffusionMap
import numpy as np


class LoadedDiffusionMap(DiffusionMap):
    def __init__(
        self,
        timespaceDomain: TimespaceDomain,
        loadPath: str,
    ):
        super().__init__(timespaceDomain)
        self.D = torch.load(loadPath)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        self.D = self.D.to(x.device)
        border = 0.1
        xc = x.ceil().long()
        xf = x.floor().long()
        xp = x - xf
        wxf = torch.where(xp <= border, 1, torch.where(xp >= 1 - border, 0, 1 - (xp - border) / (1 - 2 * border)))
        wxc = 1 - wxf

        yc = y.ceil().long()
        yf = y.floor().long()
        yp = y - yf
        wyf = torch.where(yp <= border, 1, torch.where(yp >= 1 - border, 0, 1 - (yp - border) / (1 - 2 * border)))
        wyc = 1 - wyf
        return (wxf * wyf) * self.D[xf, yf] + (wxc * wyf) * self.D[xc, yf] + (wxf * wyc) * self.D[xf, yc] + (wxc * wyc) * self.D[xc, yc]
