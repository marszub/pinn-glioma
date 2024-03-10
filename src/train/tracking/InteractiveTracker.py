from copy import deepcopy
from model.Pinn import PINN
from train.Saver import Saver
from train.tracking.SharedData import SharedData
from train.tracking.Tracker import Tracker


class InteractiveTracker(Tracker):
    def __init__(
        self, modelSaver: Saver, epochs: int, epoch: int, lossValues: list, sharedData: SharedData
    ):
        super().__init__(epochs, epoch, lossValues)
        self.modelSaver = modelSaver
        self.sharedData = sharedData

    def update(self, lossValue: tuple, nn: PINN, optimizer):
        super().update(lossValue, nn, optimizer)

        if (
            self.sharedData.unsupportedInput
            and not self.sharedData.terminate
        ):
            print("s      save model")
            print("e      exit with saving")
            print("ctrl+c exit without saving")
        
        self.sharedData.unsupportedInput = False

        if self.sharedData.save or self.epoch == self.maxEpochs:
            self.sharedData.save = False
            self.modelSaver.saveTraining(self.epoch, self.bestApprox, nn, optimizer, self.lossValues)
            self.modelSaver.saveEvalModel(self.bestApprox)
            self.modelSaver.saveMetrics(self.lossValues)
            print("Model saved")

    def isTraining(self):
        return super().isTraining() and not self.sharedData.terminate