from copy import deepcopy
from time import time
from typing import Callable
from Pinn import PINN
from train.ModelSaver import ModelSaver
from train.tracking.SharedData import SharedData
from train.tracking.Tracker import Tracker


class DefaultTracker(Tracker):
    def __init__(self, modelSaver: ModelSaver, epochs: int, sharedData: SharedData):
        super().__init__(modelSaver, epochs, sharedData)
        self.lossValues = []
        self.bestLoss = [float("inf") for _ in range(4)]

    def start(self, initialCondition: Callable, nn: PINN):
        super().start(initialCondition, nn)
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
            self.epochStartTime = time()

        if totalLoss < self.bestLoss[0]:
            self.bestApprox = deepcopy(nn)
            self.bestLoss = lossValue

        super().lateUpdate()

    # def __plotLoss(self):
    #     pass
    #     # losses = np.array(self.lossValues)
    #     # self.visualizer.plotLosses(losses[:, 0], fileName="loss_total.png")
    #     # self.visualizer.plotLosses(
    #     #     losses[:, 1:],
    #     #     labels=["Residual", "Initial", "Boundary"],
    #     #     fileName="loss_components.png",
    #     # )
