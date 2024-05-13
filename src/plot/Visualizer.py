from os import makedirs
from typing import Callable
import numpy as np
import torch
from pinn.Pinn import PINN
import matplotlib.pyplot as plt
from torch import full_like

from pinn.simulationSpace.UniformSpace import UniformSpace
from plot.Plotter import Plotter


class Visualizer:
    def __init__(
        self,
        plotter: Plotter,
        plotSpace: UniformSpace,
        saveDir: str,
        transparent: bool = True,
    ):
        makedirs(saveDir, exist_ok=True)
        self.plotter = plotter
        self.space = plotSpace
        self.saveDir = saveDir
        self.transparent = transparent

    def plotIC(self, initialCondition: Callable, title: str, name: str, diffusionMap: Callable = None):
        x, y, _ = self.space.getInitialPointsKeepDims()
        z = initialCondition(x, y)
        if diffusionMap is None:
            fig = self.plotter.plot(
                z,
                x,
                y,
                title,
            )
        else:
            D = diffusionMap(x, y)
            fig = self.plotter.plotWithBackground(
                z,
                D,
                x,
                y,
                title,
            )
        plt.figure(fig)
        plt.savefig(
            f"{self.saveDir}/{name}.png",
            transparent=self.transparent,
        )

    def plotLosses(self, loss_over_time, fileName, labels=[]):
        average_loss = self.__runningAverrage(loss_over_time, window=100)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_title("Loss function (runnig average)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(average_loss, label=labels)
        ax.set_yscale("log")
        if len(labels) > 0:
            plt.legend()
        plt.savefig(
            f"{self.saveDir}/{fileName}", transparent=self.transparent
        )
        plt.close()

    def plotLossMinMax(self, lossOverTime, fileName):
        intervalsNum = 200
        intervals = self.__splitIntervals(lossOverTime, intervalsNum)
        intervalSize = intervals.shape[1]
        x = np.arange(0, intervalSize * intervalsNum, intervalSize)
        avgLoss = np.sum(intervals, axis=1).reshape(-1) / intervalSize
        maxLoss = np.max(intervals, axis=1).reshape(-1)
        minLoss = np.min(intervals, axis=1).reshape(-1)
        globalMinX = np.argmin(minLoss)
        globalMinY = np.min(minLoss)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_title("Loss function (averrage in intervals)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(x, avgLoss, label="Averrage loss")
        ax.fill_between(x, minLoss, maxLoss, label="Min and Max in interval", alpha=0.2)
        ax.plot(globalMinX*intervalSize, globalMinY, "r.", markersize=12, label="Best fit")
        ax.set_yscale("log")
        plt.legend()
        plt.savefig(
            f"{self.saveDir}/{fileName}", transparent=self.transparent
        )
        plt.close()

    def plotSizeOverTime(
        self, pinn: PINN, name: str
    ):
        x, y, t = self.space.getInteriorPoints()
        u = pinn(x, y, t)
        uniqueT, indexes = torch.unique(t, return_inverse=True)
        sumOverTime = torch.zeros((uniqueT.size()[0], 1)).scatter_add_(0, indexes, u)
        xSpaceSize = (
            self.space.timespaceDomain.spaceDomains[0][1]
            - self.space.timespaceDomain.spaceDomains[0][0]
        )
        xpointsPerLengthUnit = self.space.spaceResoultion / xSpaceSize
        ySpaceSize = (
            self.space.timespaceDomain.spaceDomains[1][1]
            - self.space.timespaceDomain.spaceDomains[1][0]
        )
        ypointsPerLengthUnit = self.space.spaceResoultion / ySpaceSize
        pointsPerSpaceUnit = ypointsPerLengthUnit * xpointsPerLengthUnit
        intencityOverTime = sumOverTime / pointsPerSpaceUnit

        uniqueT = uniqueT.detach()
        intencityOverTime = intencityOverTime.detach()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.set_title(name)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Tumor size")
        ax.plot(uniqueT, intencityOverTime)
        plt.savefig(
            f"{self.saveDir}/{''.join(letter for letter in name if letter.isalnum())}.png", transparent=self.transparent
        )
        plt.close()

    def plotTreatment(self, treatment: Callable, name: str):
        t = torch.linspace(self.space.timespaceDomain.timeDomain[0], self.space.timespaceDomain.timeDomain[1], 500)
        f = treatment(0, 0, t)

        plt.figure(figsize=(20, 6))
        plt.plot(t, f)
        plt.xlabel('Time in days')
        plt.ylabel('Therapy factor')
        plt.title(name)
        plt.grid(True)
        plt.savefig(
            f"{self.saveDir}/{name}.png", transparent=self.transparent
        )
        plt.close()

    def animateProgress(self, pinn: PINN, fileName: str, diffusion: Callable = None):
        frameFileNames = []
        for i in range(self.space.timeResoultion + 1):
            x, y, _ = self.space.getInitialPointsKeepDims()
            timeValue = (
                self.space.timespaceDomain.timeDomain[0]
                + i
                * (
                    self.space.timespaceDomain.timeDomain[1]
                    - self.space.timespaceDomain.timeDomain[0]
                )
                / self.space.timeResoultion
            )
            t = full_like(
                x,
                timeValue,
            )
            u = pinn(x, y, t)
            frameName = self.saveDir + f"/{fileName}_{i}.png"
            if diffusion is None:
                fig = self.plotter.plot(
                    u,
                    x,
                    y,
                    f"PINN (t={timeValue:0,.2f})",
                )
            else:
                D = diffusion(x, y)
                fig = self.plotter.plotWithBackground(
                    u,
                    D,
                    x,
                    y,
                    f"PINN (t={timeValue:0,.2f})",
                )
            frameFileNames.append(frameName)
            plt.savefig(
                frameName,
                transparent=self.transparent,
                facecolor="white",
            )
            plt.close()
            self.__makeGif(
                frameFileNames=frameFileNames, fileName=fileName
            )

    def __makeGif(self, frameFileNames, fileName):
        try:
            from PIL import Image

            frames = []
            for file in frameFileNames:
                image = Image.open(file)
                frames.append(image)
            frames[0].save(
                self.saveDir + f"/{fileName}.gif",
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=500,
                loop=0,
            )
        except ImportError:
            print("Can't make gif. PIL not installed")

    def __runningAverrage(self, y, window=100):
        cumsum = np.cumsum(y, axis=0)
        return (cumsum[window:] - cumsum[:-window]) / float(window)

    def __splitIntervals(self, y, intervalsNum=500):
        intervalSize = np.floor(y.shape[0]/intervalsNum).astype(int)
        truncatedShape = list(y.shape)
        truncatedShape[0] = intervalSize * intervalsNum
        y = np.resize(y, truncatedShape)
        batchedShape = tuple([intervalsNum, intervalSize] + list(y.shape)[1:])
        return np.reshape(y, batchedShape)
