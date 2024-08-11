from typing import Tuple, Callable, NamedTuple

from pinn.Pinn import PINN


class Loss(NamedTuple):
    residual: Tuple[float, Callable]
    initial: Tuple[float, Callable]
    boundary: Tuple[float, Callable]
    data: Tuple[float, Callable]
    validation: Callable

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function
        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual[0] * self.residual[1](pinn)
        initial_loss = self.initial[0] * self.initial[1](pinn)
        boundary_loss = self.boundary[0] * self.boundary[1](pinn)
        data_loss = self.data[0] * self.data[1](pinn)

        final_loss = (
            residual_loss
            + initial_loss
            + boundary_loss
            + data_loss
        )

        return (
            final_loss,
            residual_loss,
            initial_loss,
            boundary_loss,
            data_loss
        )

    def validate(self, pinn: PINN):
        return self.verbose(pinn) + (self.validation(pinn),)

    def __call__(self, pinn: PINN):
        """
        Allows you to use instance of this class as if it was a function:
        ```
        >>> loss = Loss(*some_args)
        >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)[0]

    def to(self, device):
        self.residual[1].to(device)
        self.initial[1].to(device)
        self.boundary[1].to(device)
        self.data[1].to(device)
        self.validation.to(device)
        return self
