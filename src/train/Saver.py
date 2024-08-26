from os import makedirs
from torch import save


class Saver:
    def __init__(self, saveDir: str):
        makedirs(saveDir, exist_ok=True)
        self.saveDir = saveDir
        self.evalModelPath = self.saveDir + "/model_best.pt"
        self.trainStatePath = self.saveDir + "/training_state.pt"
        self.metricsPath = self.saveDir + "/loss_over_time.txt"

    def saveEvalModel(self, model):
        device = model.device()
        save_dict = {
            "layers": model.layers,
            "neurons": model.neurons,
            "model": model.cpu().state_dict(),
        }
        save(save_dict, self.evalModelPath)
        model.to(device)

    def saveTraining(
        self, epoch, bestModel, model, optimizer, lossOverTime
    ):
        device = model.device()
        save(
            {
                "epoch": epoch,
                "layers": model.layers,
                "neurons": model.neurons,
                "bestModel": bestModel.cpu().state_dict(),
                "model": model.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "lossOverTime": lossOverTime,
            },
            self.trainStatePath,
        )
        model.to(device)
        bestModel.to(device)

    def saveMetrics(self, lossOverTime):
        with open(self.metricsPath, "w") as file:
            file.writelines(
                " ".join(str(j) for j in i) + "\n" for i in lossOverTime
            )
