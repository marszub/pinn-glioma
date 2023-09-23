from copy import deepcopy
from time import time
from typing import Callable
import numpy as np
from Pinn import PINN

from tracking.Tracker import Tracker
from tracking.Visualizer import Visualizer


class SimpleTracker(Tracker):
    def __init__(self, visualizer: Visualizer, epochs: int, isInteractive: bool, startPaused: bool):
        super().__init__(visualizer, epochs, isInteractive, startPaused)
        self.lossValues = []
        self.bestLoss = [float("inf") for _ in range(4)]

    def start(self, initialCondition: Callable, nn: PINN):
        super().start(initialCondition, nn)
        self.startTime = time()

    def update(self, lossValue: tuple, nn: PINN):
        super().update(lossValue, nn)
        lossValue = list(map(lambda element: element.item(), lossValue))
        self.lossValues.append(lossValue)

        if lossValue[0] < self.bestLoss[0]:
            self.bestApprox = deepcopy(nn)
            self.bestLoss = lossValue

    def finish(self, pinn: PINN):
        self.visualizer.saveModel(self.bestApprox)
        losses = np.array(self.lossValues)
        self.visualizer.plotLosses(losses[:, 0], fileName="loss_total.png")
        self.visualizer.printAverageTime((time() - self.startTime) / self.epoch)
