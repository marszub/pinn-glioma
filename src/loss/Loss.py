from typing import Callable
from loss.Treatment import Treatment
from simulationSpace.SampleSpace import SampleSpace
from Pinn import PINN
from Weights import Weights
from loss.Function import *

class Loss:
    def __init__(
        self,
        space: SampleSpace,
        initialCondition: Callable,
        diffusion: Callable,
        treatment: Treatment,
        weights: Weights,
    ):
        self.space = space
        self.initialCondition = initialCondition
        self.diffusion = diffusion
        self.treatment = treatment
        self.weighs = weights

    def __residual(self, pinn: PINN):
        x, y, t = self.space.getInteriorPoints()
        rho = 0.025

        D = self.diffusion(x, y)
        R = self.treatment(x, y, t)
        u = f(pinn, x, y, t)
        loss = (
            dfdt(pinn, x, y, t)
            - D * dfdx(pinn, x, y, t, order=2)
            - D * dfdy(pinn, x, y, t, order=2)
            - rho * u * (1 - u)
            + R * u
        )
        return loss.pow(2).mean()

    def __initial(self, pinn: PINN):
        x, y, t = self.space.getInitialPoints()
        pinn_init = self.initialCondition(x, y)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def __boundary(self, pinn: PINN):
        down, up, left, right = self.space.getBoundaryPoints()
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
        self.space.to(pinn.device())
        residual_loss = self.__residual(pinn)
        initial_loss = self.__initial(pinn)
        boundary_loss = self.__boundary(pinn)

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