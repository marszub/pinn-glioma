from model.Pinn import PINN
from train.Saver import Saver
from train.tracking.Tracker import Tracker


class AutosaveTracker(Tracker):
    def __init__(self, modelSaver: Saver, epochs: int):
        super().__init__(modelSaver, epochs)

    def update(self, lossValue: tuple, nn: PINN):
        super().update(lossValue, nn)
        if self.epoch % 10000 or self.isTerminated():
            self.modelSaver.saveModel(self.bestApprox)
            print("Model saved")
