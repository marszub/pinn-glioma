from model.LoadedInitialCondition import LoadedInitialCondition
from model.LoadedDiffusionMap import LoadedDiffusionMap
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
            absorptionRate=0.0,
            decayRate=0.0,
            dose=0.0,
            firstDoseTime=0.0,
            dosesNum=0,
            timeBetweenDoses=0.0,
        )
        self.ic = LoadedInitialCondition(
            self.timespaceDomain, "resources/ic2.pt")
