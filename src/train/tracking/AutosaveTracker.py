from model.Pinn import PINN
from train.Saver import Saver
from train.tracking.Tracker import Tracker


class AutosaveTracker(Tracker):
    def __init__(self, modelSaver: Saver, epochs: int, epoch:int, lossValues: list):
        super().__init__(modelSaver, epochs, epoch, lossValues)
        self.modelSaver = modelSaver

    def update(self, lossValue: tuple, nn: PINN, optimizer):
        super().update(lossValue, nn, optimizer)
        if self.epoch % 10000 == 0 or not self.isTraining():
            self.modelSaver.saveTraining(self.epoch, self.bestApprox, nn, optimizer, self.lossValues)
            self.modelSaver.saveEvalModel(self.bestApprox)
            self.modelSaver.saveMetrics(self.lossValues)
            print("Model saved")
