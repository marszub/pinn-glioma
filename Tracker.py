from copy import deepcopy
from os import makedirs
from time import time
from typing import Callable

import numpy as np
from Pinn import PINN
from SampleSpace import SampleSpace
from Visualize import animate_progress, plot_initial_condition, plot_losses, write_loss
from torch import save

class Tracker:
    def __init__(self, logDir: str, plotSpace: SampleSpace):
        makedirs(logDir, exist_ok=True)
        self.logDir = logDir
        self.space = plotSpace
        self.lossValues = []
        self.bestLoss = float('inf')

    def start(self, initialCondition: Callable):
        plot_initial_condition(self.space, initialCondition)
        self.epochStartTime = time()
        self.epoch = 0

    def update(self, lossValue: tuple, nn: PINN):
        lossValue = list(map(lambda element: element.item(), lossValue))
        self.lossValues.append(lossValue)

        totalLoss = lossValue[0]
        if (self.epoch + 1) % 1000 == 0:
            print(
                f"Epoch: {self.epoch + 1}\tLoss: {float(totalLoss):>7f}\t"
                + f"Time: {time() - self.epochStartTime:.1f}s"
            )
            self.plotLoss()
            self.epochStartTime = time()

        if totalLoss < self.bestLoss:
            self.bestApprox = deepcopy(nn)
            self.bestLoss = totalLoss
        self.epoch += 1

    def finish(self, pinn: PINN):
        pinn = pinn.cpu()
        save(pinn.state_dict(), self.logDir + "/model")
        write_loss(self.lossValues[-1])
        self.plotLoss()
        animate_progress(pinn, self.space)

    def plotLoss(self):
        losses = np.array(self.lossValues)
        plot_losses(losses[:, 0], filePath=self.logDir + "/loss_total.png")
        plot_losses(losses[:, 1:], labels=["Residual", "Initial", "Boundary"], filePath=self.logDir + "/loss_components.png")