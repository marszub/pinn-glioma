from os import makedirs
from typing import Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from plot.DataProvider import DataProvider
from model.TimespaceDomain import TimespaceDomain

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

    def plotIC(
        self,
        initialCondition: Callable,
        title: str,
        name: str,
        diffusionMap: Callable = None
    ):
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
        ax.fill_between(x, minLoss, maxLoss,
                        label="Min and Max in interval", alpha=0.2)
        ax.plot(globalMinX*intervalSize, globalMinY,
                "r.", markersize=12, label="Best fit")
        ax.set_yscale("log")
        plt.legend()
        plt.savefig(
            f"{self.saveDir}/{fileName}", transparent=self.transparent
        )
        plt.close()

    def plotSizeOverTime(
        self, times: Tensor, sizes: Tensor, filename: str, title: str
    ):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.set_title(title)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel("Tumor size")
        ax.plot(times, sizes)
        plt.savefig(
            f"{self.saveDir}/{''.join(letter for letter in filename if letter.isalnum())}.png",
            transparent=self.transparent
        )
        plt.close()

    def plotTreatment(self, treatment: Callable, name: str):
        t = torch.linspace(
            self.space.timespaceDomain.timeDomain[0],
            self.space.timespaceDomain.timeDomain[1],
            500
        )
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

    def animateProgress(
        self,
        data_provider: DataProvider,
        fileName: str,
        diffusion: Callable = None
    ):
        class Iteration:
            def __init__(
                self,
                save_dir: str,
                plotter: Plotter,
                transparent: bool,
                space: TimespaceDomain,
                diffusion: Callable
            ):
                self.save_dir = save_dir
                self.plotter = plotter
                self.transparent = transparent
                self.space = space
                self.diffusion = diffusion
                self.i = 0
                self.frame_file_names = []

            def __call__(self, t, u):
                assert u.shape[0] == u.shape[1]
                spatial_resolution = u.shape[0]
                frameName = self.save_dir + f"/{fileName}_{self.i}.png"
                space = UniformSpace(self.space, spatial_resolution, 0)
                x, y, _ = space.getInitialPointsKeepDims()
                x = x.squeeze(axis=-1)
                y = y.squeeze(axis=-1)
                if self.diffusion is None:
                    _ = self.plotter.plot(
                        u,
                        x,
                        y,
                        f"PINN (t={t:4.2f})",
                    )
                else:
                    D = self.diffusion(x, y)
                    _ = self.plotter.plotWithBackground(
                        u,
                        D,
                        x,
                        y,
                        f"PINN (t={t:0,.2f})",
                    )
                self.frame_file_names.append(frameName)
                plt.savefig(
                    frameName,
                    transparent=self.transparent,
                    facecolor="white",
                )
                plt.close()
                self.i += 1

        iteration = Iteration(
            self.saveDir,
            self.plotter,
            self.transparent,
            data_provider.timespace_domain,
            diffusion
        )

        data_provider.for_each_frame(iteration)
        self.__makeGif(
            frameFileNames=iteration.frame_file_names, fileName=fileName
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
