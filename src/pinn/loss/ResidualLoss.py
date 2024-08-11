from model.Experiment import Experiment
from pinn.sample.SampleSpace import SampleSpace
from pinn.Pinn import PINN
from pinn.loss.Function import f, dfdt, dfdx, dfdy


class ResidualLoss:
    def __init__(self, experiment: Experiment, sample_space: SampleSpace):
        self.experiment = experiment
        self.sample_space = sample_space

    def __call__(self, pinn: PINN):
        x, y, t = self.sample_space.get_points()
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

    def to(self, device):
        self.sample_space.to(device)
        return self
