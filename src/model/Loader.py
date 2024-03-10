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
    bestModel.eval()
    state["bestModel"] = bestModel

    model.load_state_dict(state["model"])
    model.eval()
    state["model"] = model

    optimizer.load_state_dict(state["optimizer"])
    state["optimizer"] = optimizer

    return state


def loadMetrics(path):
    if not os.path.isfile(path):
        return None
    lossOverTime = []
    with open(path, "r") as f:
        for line in f.readlines():
            lossOverTime.append(line.split(" "))
    return lossOverTime
