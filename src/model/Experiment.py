from model.InitialCondition import InitialCondition
from model.LoadedDiffusionMap import LoadedDiffusionMap
# from model.LoadedInitialCondition import LoadedInitialCondition
from model.Treatment import Treatment
from model.TimespaceDomain import TimespaceDomain


class Experiment:
    def __init__(self):
        self.rho = 0.025
        self.timespaceDomain = TimespaceDomain(
            spaceDomains=[(0.0, 50.0), (0.0, 80.0)],
            timeDomain=(0.0, 100.0),
        )
        self.diffusion = LoadedDiffusionMap(
            self.timespaceDomain,
            "resources/diffusion_map.pt"
        )
        self.treatment = Treatment(
            absorptionRate=0.5,
            decayRate=0.02,
            dose=0.05,
            firstDoseTime=50.0,
            dosesNum=2,
            timeBetweenDoses=20.0,
        )
        self.ic = InitialCondition((28.0, 45.0), 0.4, 10)
        # self.ic = LoadedInitialCondition(self.timespaceDomain, "resources/diffusion_map.pt")
