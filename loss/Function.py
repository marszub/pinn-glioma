import torch
from Pinn import PINN
from torch import Tensor


def f(pinn: PINN, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
    """Compute the value of the approximate solution from the NN model"""
    return pinn(x, y, t)


def df(output: Tensor, input: Tensor, order: int = 1) -> Tensor:
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
