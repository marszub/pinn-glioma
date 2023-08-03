from typing import Tuple
from typing import Callable
from SampleSpace import SampleSpace
import torch
from torch import Tensor
from Pinn import PINN
from Samples import get_initial_points, get_interior_points, get_boundary_points
from Weights import Weights

def f(
    pinn: PINN, x: Tensor, y: Tensor, t: Tensor
) -> Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y, t)


def df(
    output: Tensor, input: Tensor, order: int = 1
) -> Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine"""
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value


def dfdt(
    pinn: PINN,
    x: Tensor,
    y: Tensor,
    t: Tensor,
    order: int = 1,
):
    f_value = f(pinn, x, y, t)
    return df(f_value, t, order=order)


def dfdx(
    pinn: PINN,
    x: Tensor,
    y: Tensor,
    t: Tensor,
    order: int = 1,
):
    f_value = f(pinn, x, y, t)
    return df(f_value, x, order=order)


def dfdy(
    pinn: PINN,
    x: Tensor,
    y: Tensor,
    t: Tensor,
    order: int = 1,
):
    f_value = f(pinn, x, y, t)
    return df(f_value, y, order=order)


class Loss:
    def __init__(
        self,
        space: SampleSpace,
        initial_condition: Callable,
        weights: Weights,
        verbose: bool = False,
    ):
        self.space = space
        self.initial_condition = initial_condition
        self.weighs = weights

    def residual_loss(self, pinn: PINN):
        x, y, t = get_interior_points(
            self.space,
            pinn.device(),
        )
        rho = 0.025

        def D_fun(x, y) -> Tensor:
            res = torch.zeros(x.shape, dtype=x.dtype, device=pinn.device())
            dist = (x - 0.5) ** 2 + (y - 0.5) ** 2
            res[dist < 0.25] = 0.13
            res[dist < 0.02] = 0.013
            return res

        # def phi(x, y) -> torch.Tensor:
        #     res = torch.ones(x.shape, dtype=x.dtype, device=pinn.device())
        #     dist = (x-0.5)**2 + (y-0.5)**2
        #     res[dist >= 0.25] = 0
        #     return res
        D = D_fun(x, y)
        # mask = D > 0
        u = f(pinn, x, y, t)
        # Phi = phi(x, y)
        loss = (
            dfdt(pinn, x, y, t)
            - D * dfdx(pinn, x, y, t, order=2)
            - D * dfdy(pinn, x, y, t, order=2)
            - rho * u * (1 - u)
        )
        # loss = mask * loss + torch.logical_not(mask) * u
        return loss.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        x, y, t = get_initial_points(
            self.space,
            pinn.device(),
        )
        pinn_init = self.initial_condition(x, y)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = get_boundary_points(
            self.space,
            pinn.device(),
        )
        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right
        loss_down = dfdy(pinn, x_down, y_down, t_down)
        loss_up = dfdy(pinn, x_up, y_up, t_up)
        loss_left = dfdx(pinn, x_left, y_left, t_left)
        loss_right = dfdx(pinn, x_right, y_right, t_right)

        return (
            loss_down.pow(2).mean()
            + loss_up.pow(2).mean()
            + loss_left.pow(2).mean()
            + loss_right.pow(2).mean()
        )

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function
        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = (
            self.weighs.residual * residual_loss
            + self.weighs.initial * initial_loss
            + self.weighs.boundary * boundary_loss
        )

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use instance of this class as if it was a function:
        ```
        >>> loss = Loss(*some_args)
        >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)[0]