from copy import deepcopy
from model.Pinn import PINN
from train.Saver import Saver
from train.tracking.SharedData import SharedData
from train.tracking.Tracker import Tracker


class InteractiveTracker(Tracker):
    def __init__(
        self, modelSaver: Saver, epochs: int, sharedData: SharedData
    ):
        super().__init__(modelSaver, epochs)
        self.sharedData = sharedData

    def update(self, lossValue: tuple, nn: PINN):
        super().update(lossValue, nn)

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
            self.modelSaver.saveEvalModel(self.bestApprox)
            print("Model saved")
