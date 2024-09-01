from pinn.loss.Loss import Loss
from model.Experiment import Experiment


class PinnConfig:
    def __init__(
        self,
        experiment: Experiment,
        data_dir: str = None,
        validation_dir: str = None,
    ):
        from pinn.loss.SampleSizes import SampleSizes
        from pinn.sample.interior.DataFocusedRandom import DataFocusedRandom
        from pinn.loss.ResidualLoss import ResidualLoss
        from pinn.loss.BoundaryLoss import BoundaryLoss
        from pinn.loss.DataLoss import DataLoss
        from pinn.loss.ValidationLoss import ValidationLoss
        from pinn.loss.ZeroLoss import ZeroLoss

        sample_sizes = SampleSizes.scaled(32)
        initial_loss = ZeroLoss()

        # from pinn.loss.InitialLoss import InitialLoss
        # initial_loss = InitialLoss(
        #     experiment.timespaceDomain,
        #     sample_sizes.initial,
        #     experiment.ic,
        # )
        boundary_loss = BoundaryLoss(
            experiment.timespaceDomain,
            sample_sizes.boundary,
        )
        data_loss = DataLoss(
            experiment.timespaceDomain,
            sample_sizes.data,
            data_dir,
        )
        interior_sample = DataFocusedRandom(
            timespace_domain=experiment.timespaceDomain,
            sample_size=sample_sizes.interior,
            times=data_loss.times,
            rate=0.2,
        )
        print(data_loss.times)
        residual_loss = ResidualLoss(experiment, interior_sample)
        validation_loss = ValidationLoss(
            experiment.timespaceDomain,
            samples_num=sample_sizes.validation,
            data_dir_name=validation_dir,
        )
        self.loss = Loss(
            residual=(2.0, residual_loss),
            initial=(1.0, initial_loss),
            boundary=(1.0, boundary_loss),
            data=(1.0, data_loss),
            validation=validation_loss,
        )
