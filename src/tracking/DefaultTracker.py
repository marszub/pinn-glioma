from copy import deepcopy
from time import time
from typing import Callable
import numpy as np
from Pinn import PINN

from tracking.Tracker import Tracker
from tracking.Visualizer import Visualizer


class DefaultTracker(Tracker):
    def __init__(self, visualizer: Visualizer, epochs: int, isInteractive: bool, startPaused: bool):
        super().__init__(visualizer, epochs, isInteractive, startPaused)
        self.lossValues = []
        self.bestLoss = [float("inf") for _ in range(4)]

    def start(self, initialCondition: Callable, nn: PINN):
        super().start(initialCondition, nn)
        self.visualizer.plotIC(initialCondition)
        self.epochStartTime = time()

    def update(self, lossValue: tuple, nn: PINN):
        super().update(lossValue, nn)
        lossValue = list(map(lambda element: element.item(), lossValue))
        self.lossValues.append(lossValue)

        totalLoss = lossValue[0]
        if (self.epoch) % 1000 == 0:
            print(
                f"Epoch: {self.epoch}\tLoss: {float(totalLoss):>7f}\t"
                + f"Time: {time() - self.epochStartTime:.1f}s"
            )
            self.__plotLoss()
            self.epochStartTime = time()

        if totalLoss < self.bestLoss[0]:
            self.bestApprox = deepcopy(nn)
            self.bestLoss = lossValue

    def finish(self, pinn: PINN):
        self.visualizer.saveModel(self.bestApprox)
        self.visualizer.printLoss(self.bestLoss)
        self.__plotLoss()
        self.visualizer.animateProgress(self.bestApprox.cpu(), "animation")

    def __plotLoss(self):
        losses = np.array(self.lossValues)
        self.visualizer.plotLosses(losses[:, 0], fileName="loss_total.png")
        self.visualizer.plotLosses(
            losses[:, 1:],
            labels=["Residual", "Initial", "Boundary"],
            fileName="loss_components.png",
        )
