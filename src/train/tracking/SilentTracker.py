from copy import deepcopy
from time import time
from model.Pinn import PINN
from train.tracking import SharedData
from train.tracking.Tracker import Tracker
from train.ModelSaver import ModelSaver


class SilentTracker(Tracker):
    def __init__(
        self, modelSaver: ModelSaver, epochs: int, sharedData: SharedData
    ):
        super().__init__(modelSaver, epochs, sharedData)
        self.lossValues = []
        self.bestLoss = [float("inf") for _ in range(4)]

    def start(self, nn: PINN):
        super().start(nn)
        self.startTime = time()

    def update(self, lossValue: tuple, nn: PINN):
        super().update(lossValue, nn)
        lossValue = list(map(lambda element: element.item(), lossValue))
        self.lossValues.append(lossValue)

        if lossValue[0] < self.bestLoss[0]:
            self.bestApprox = deepcopy(nn)
            self.bestLoss = lossValue

        super().lateUpdate()
