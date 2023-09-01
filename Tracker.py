from copy import deepcopy
from os import makedirs
from time import time
from typing import Callable

import numpy as np
from Pinn import PINN
from Visualize import animate_progress, plot_initial_condition, plot_losses, write_loss
from torch import save

from simulationSpace.UniformSpace import UniformSpace

class Tracker:
    def __init__(self, logDir: str, plotSpace: UniformSpace):
        makedirs(logDir, exist_ok=True)
        self.logDir = logDir
        self.space = plotSpace
        self.lossValues = []
        self.bestLoss = float('inf')

    def start(self, initialCondition: Callable):
        plot_initial_condition(self.space, initialCondition, self.logDir + "/initial_condition.png")
        self.epochStartTime = time()
        self.epoch = 0

    def update(self, lossValue: tuple, nn: PINN):
        self.epoch += 1
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

        if totalLoss < self.bestLoss:
            self.bestApprox = deepcopy(nn)
            self.bestLoss = totalLoss

    def finish(self, pinn: PINN):
        bestApprox = self.bestApprox.cpu()
        save(bestApprox.state_dict(), self.logDir + "/model")
        write_loss(self.lossValues[-1])
        self.__plotLoss()
        self.animate()

    def animate(self):
        animate_progress(self.bestApprox.cpu(), self.space, self.logDir)

    def __plotLoss(self):
        losses = np.array(self.lossValues)
        plot_losses(losses[:, 0], filePath=self.logDir + "/loss_total.png")
        plot_losses(losses[:, 1:], labels=["Residual", "Initial", "Boundary"], filePath=self.logDir + "/loss_components.png")
