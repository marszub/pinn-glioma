import torch
from os import path

from model.TimespaceDomain import TimespaceDomain
from pinn.Pinn import PINN
from model.DataLoader import DataLoader


class ValidationLoss:
    def __init__(
        self,
        timespace_domain: TimespaceDomain,
        data_dir_name: str = None,
    ):
        if data_dir_name is None:
            self.is_data = False
            return

        assert path.isdir(data_dir_name)
        data_loader = DataLoader(data_dir_name)
        indices = data_loader.get_indices()
        assert len(indices) > 1

        u, t_linspace = data_loader.get_data(indices)

        x_linspace = torch.linspace(
            timespace_domain.spaceDomains[0][0],
            timespace_domain.spaceDomains[0][1],
            u.shape[0],
        )
        y_linspace = torch.linspace(
            timespace_domain.spaceDomains[1][0],
            timespace_domain.spaceDomains[1][1],
            u.shape[1],
        )
        inputs = torch.meshgrid(
            x_linspace,
            y_linspace,
            t_linspace,
            indexing="ij")
        self.input = torch.stack(inputs, axis=-1).reshape((-1, 3))
        self.gt = u.reshape((-1, 1))
        self.device = torch.device("cpu")
        # is_data indicates if member tensors exist.
        # If not, it means no data set was provided.
        self.is_data = True

    def __call__(self, pinn: PINN):
        if not self.is_data:
            return torch.tensor(0.0, device=self.device)
        output = pinn(
            self.input[:, 0:1],
            self.input[:, 1:2],
            self.input[:, 2:3],
        )
        return (self.gt - output).pow(2).mean()

    def to(self, device):
        self.device = device
        if not self.is_data:
            return self
        self.input = self.input.to(device)
        self.gt = self.gt.to(device)
        return self
