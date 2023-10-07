from os import makedirs

from torch import save


class ModelSaver:
    def __init__(self, saveDir: str):
        makedirs(saveDir, exist_ok=True)
        self.saveDir = saveDir

    def saveModel(self, model):
        save(model.cpu().state_dict(), self.saveDir + "/model")
