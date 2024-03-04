from os import makedirs

from torch import save


class Saver:
    def __init__(self, saveDir: str):
        makedirs(saveDir, exist_ok=True)
        self.saveDir = saveDir
        self.evalModelPath = self.saveDir + "/model_best.pt"
        self.trainStatePath = self.saveDir + "/training_state.pt"

    def saveEvalModel(self, model):
        save(model.cpu().state_dict(), self.evalModelPath)

    def saveTraining(self, bestModel, model, optimizer):
        save(
            {
                "bestModel": bestModel.cpu().state_dict(),
                "model": model.cpu().state_dict(),
                "optimizer": optimizer.cpu().state_dict(),
            },
            self.trainStatePath,
        )

    def saveMetrics(self, lossOverTime):
        pass
