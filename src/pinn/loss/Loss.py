from model.Experiment import Experiment
from pinn.simulationSpace.SampleSpace import SampleSpace
from pinn.Pinn import PINN
from pinn.loss.Weights import Weights
from pinn.loss.Function import f, dfdx, dfdy, dfdt
from pinn.loss.DataLoss import DataLoss


class Loss:
    def __init__(
        self,
        space: SampleSpace,
        experiment: Experiment,
        data_dir_name: str,
        data_samples_num: int,
        weights: Weights,
    ):
        self.space = space
        self.experiment = experiment
        self.weighs = weights
        self.data_loss = DataLoss(
            experiment.timespaceDomain, data_samples_num, data_dir_name)

    def __residual(self, pinn: PINN):
        x, y, t = self.space.getInteriorPoints()
        rho = self.experiment.rho

        D = self.experiment.diffusion(x, y)
        R = self.experiment.treatment(x, y, t)
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
        pinn_init = self.experiment.ic(x, y)
        loss = f(pinn, x, y, t) - pinn_init
        return loss.pow(2).mean()

    def __boundary(self, pinn: PINN):
        down, up, left, right = self.space.getBoundaryPoints()
        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right
        loss_down = f(pinn, x_down, y_down, t_down)
        loss_up = f(pinn, x_up, y_up, t_up)
        loss_left = f(pinn, x_left, y_left, t_left)
        loss_right = f(pinn, x_right, y_right, t_right)

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
        self.data_loss.to(pinn.device())
        residual_loss = self.__residual(pinn)
        initial_loss = self.__initial(pinn)
        boundary_loss = self.__boundary(pinn)
        data_loss = self.data_loss(pinn)

        final_loss = (
            self.weighs.residual * residual_loss
            + self.weighs.initial * initial_loss
            + self.weighs.boundary * boundary_loss
            + self.weighs.data * data_loss
        )

        return final_loss, residual_loss, initial_loss, boundary_loss, data_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use instance of this class as if it was a function:
        ```
        >>> loss = Loss(*some_args)
        >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)[0]
