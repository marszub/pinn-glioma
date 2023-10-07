from copy import deepcopy
from time import time
from model.Pinn import PINN
from train.ModelSaver import ModelSaver
from train.tracking.SharedData import SharedData
from train.tracking.Tracker import Tracker


class DefaultTracker(Tracker):
    def __init__(
        self, modelSaver: ModelSaver, epochs: int, sharedData: SharedData
    ):
        super().__init__(modelSaver, epochs, sharedData)
        self.lossValues = []
        self.bestLoss = [float("inf") for _ in range(4)]

    def start(self, nn: PINN):
        super().start(nn)
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

        if self.sharedData.save:
            print("Saved")
        elif (
            self.sharedData.unsupportedInput
            and not self.sharedData.terminate
        ):
            print("s      save model")
            print("e      exit with saving")
            print("Ctrl+C exit without saving")
        
        self.sharedData.unsupportedInput = False

        super().lateUpdate()

