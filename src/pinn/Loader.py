import os.path
import torch


def loadModel(path):
    from pinn.Pinn import PINN
    if not os.path.isfile(path):
        return None
    model_state = torch.load(path)
    model = PINN(layers=model_state["layers"],
                 neuronsPerLayer=model_state["neurons"])
    model.load_state_dict(model_state["model"])
    model.eval()
    return model


def loadTrainState(path, device):
    if not os.path.isfile(path):
        return None
    from pinn.Pinn import PINN
    state = torch.load(path)

    bestModel = PINN(
        layers=state["layers"], neuronsPerLayer=state["neurons"]).to(device)
    bestModel.load_state_dict(state["bestModel"])
    bestModel.eval()
    state["bestModel"] = bestModel

    model = PINN(
        layers=state["layers"], neuronsPerLayer=state["neurons"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    state["model"] = model

    optimizer = torch.optim.Adam(model.parameters())
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
