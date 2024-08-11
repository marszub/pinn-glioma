from pinn.Pinn import PINN
from torch import nn
from pinn.loss.Loss import Loss
from model.Experiment import Experiment


class PinnConfig:
    def __init__(self, experiment: Experiment, data_dir: str = None, validation_dir: str = None):
        from pinn.loss.SampleSizes import SampleSizes
        from pinn.sample.interior.ExponentialRandom import ExponentialRandom
        from pinn.loss.ResidualLoss import ResidualLoss
        from pinn.loss.InitialLoss import InitialLoss
        from pinn.loss.BoundaryLoss import BoundaryLoss
        from pinn.loss.DataLoss import DataLoss
        from pinn.loss.ValidationLoss import ValidationLoss

        sample_sizes = SampleSizes.scaled(32)
        interior_sample = ExponentialRandom(
            timespace_domain=experiment.timespaceDomain,
            sample_size=sample_sizes.interior,
            rate=1.0
        )
        residual_loss = ResidualLoss(experiment, interior_sample)
        initial_loss = InitialLoss(
            experiment.timespaceDomain,
            sample_sizes.initial,
            experiment.ic,
        )
        boundary_loss = BoundaryLoss(
            experiment.timespaceDomain,
            sample_sizes.boundary,
        )
        data_loss = DataLoss(
            experiment.timespaceDomain,
            sample_sizes.data,
            data_dir,
        )
        validation_loss = ValidationLoss(
            experiment.timespaceDomain,
            validation_dir,
        )
        self.loss = Loss(
            residual=(2.0, residual_loss),
            initial=(1.0, initial_loss),
            boundary=(1.0, boundary_loss),
            data=(1.0, data_loss),
            validation=validation_loss,
        )

    def getNeuralNetwork(self) -> PINN:
        return PINN(layers=4, neuronsPerLayer=120, act=nn.Tanh())
