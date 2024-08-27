from torch import Tensor
import torch
from model.TimespaceDomain import TimespaceDomain


class LoadedInitialCondition:
    def __init__(
        self,
        timespace_domain: TimespaceDomain,
        loadPath: str,
    ):
        self.timespace_domain = timespace_domain
        self.u = torch.load(loadPath)

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        self.u = self.u.to(x.device)
        x_min, x_max = self.timespace_domain.spaceDomains[0]
        y_min, y_max = self.timespace_domain.spaceDomains[1]
        di = 0.000001
        xi_max = self.u.shape[0] - 1 - di
        xi_min = di
        yi_max = self.u.shape[1] - 1 - di
        yi_min = di

        x = (x - x_min) * (xi_max - xi_min) / (x_max - x_min) + xi_min
        y = (y - y_min) * (yi_max - yi_min) / (y_max - y_min) + yi_min

        xc = x.ceil().long()
        xf = x.floor().long()
        wxf = x - xf
        wxc = 1.0 - wxf

        yc = y.ceil().long()
        yf = y.floor().long()
        wyf = y - yf
        wyc = 1.0 - wyf
        u = ((wxf * wyf) * self.u[xf, yf] +
             (wxc * wyf) * self.u[xc, yf] +
             (wxf * wyc) * self.u[xf, yc] +
             (wxc * wyc) * self.u[xc, yc])
        return u
