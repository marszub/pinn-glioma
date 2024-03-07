from torch import load
import os.path


def loadModel(model, path):
    if not os.path.isfile(path):
        return None
    model.load_state_dict(load(path))
    model.eval()
    return model


def loadTrainState(bestModel, model, optimizer, path):
    if not os.path.isfile(path):
        return None
    state = load(path)
    bestModel.load_state_dict(state["bestModel"])
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    bestModel.eval()
    model.eval()
    return {
        "bestModel": bestModel.cpu().state_dict(),
        "trainModel": model.cpu().state_dict(),
        "optimizer": optimizer.cpu().state_dict(),
    }
    
def loadMetrics(path):
    if not os.path.isfile(path):
        return None
    lossOverTime = []
    with open(path, 'r') as f:
        for line in f.readlines():
            lossOverTime.append(line.split(' '))
    return lossOverTime