from os import makedirs
from typing import Callable
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
from torch import Tensor
from plot.DataProvider import DataProvider
from model.TimespaceDomain import TimespaceDomain

from pinn.simulationSpace.UniformSpace import UniformSpace
from plot.Plotter import Plotter


def make_title(title_lines):
    title = ""
    for title_line in title_lines:
        title += title_line + "\n"
    if len(title_lines) != 0:
        title = title[:-1]
    return title


class Visualizer:
    def __init__(
        self,
        plot_space: UniformSpace,
        file_prefix: str,
        title: list,
        transparent: bool,
    ):
        path_match = re.fullmatch(
            r"(?P<dir>/?([\w.-]+/)*)(?P<filename>[\w.-]+)", file_prefix)
        error_msg = ("File prefix must end "
                     "with non-empty file name prefix. "
                     "Moreover, allowed path characters are: a-zA-Z0-9_.-/ "
                     f"Actual prefix: {file_prefix}"
                     )
        assert path_match is not None, error_msg
        dir_path = path_match.group("dir")
        if dir_path == "":
            dir_path = "."
        if dir_path[-1] == "/" and len(dir_path) > 1:
            dir_path = dir_path[:-1]
        makedirs(dir_path, exist_ok=True)
        self.space = plot_space
        self.file_prefix = file_prefix
        self.title = title
        self.transparent = transparent

    def plotIC(
        self,
        initialCondition: Callable,
        plotter: Plotter,
        diffusionMap: Callable = None
    ):
        x, y, _ = self.space.getInitialPointsKeepDims()
        z = initialCondition(x, y)
        if diffusionMap is None:
            fig = plotter.plot(
                z,
                x,
                y,
                make_title(self.title),
            )
        else:
            D = diffusionMap(x, y)
            fig = plotter.plotWithBackground(
                z,
                D,
                x,
                y,
                make_title(self.title),
            )
        plt.figure(fig)
        plt.savefig(
            f"{self.file_prefix}.png",
            transparent=self.transparent,
        )

    def plotLosses(self, loss_over_time, labels=[]):
        average_loss = self.__runningAverrage(loss_over_time, window=100)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_title(make_title(self.title))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.plot(average_loss, label=labels)
        ax.set_yscale("log")
        if len(labels) > 0:
            plt.legend()
        plt.savefig(
            f"{self.file_prefix}.png", transparent=self.transparent
        )

    def plotLossMinMax(self, lossOverTime):
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
        ax.set_title(make_title(self.title))
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
            f"{self.file_prefix}.png", transparent=self.transparent
        )

    def plotSizeOverTime(
        self, times: Tensor, sizes: Tensor, y_title: str
    ):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.set_title(make_title(self.title))
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(y_title)
        ax.plot(times, sizes)
        plt.savefig(
            f"{self.file_prefix}.png",
            transparent=self.transparent
        )

    def plotTreatment(self, treatment: Callable):
        t = torch.linspace(
            self.space.timespaceDomain.timeDomain[0],
            self.space.timespaceDomain.timeDomain[1],
            500
        )
        f = treatment(0, 0, t)

        plt.figure(figsize=(12, 4), dpi=200)
        plt.plot(t, f)
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.title(make_title(self.title))
        plt.grid(True)
        plt.savefig(
            f"{self.file_prefix}.png", transparent=self.transparent
        )

    def animateProgress(
        self,
        data_provider: DataProvider,
        plotter: Plotter,
        diffusion: Callable = None
    ):
        class Iteration:
            def __init__(
                self,
                file_prefix: str,
                title: str,
                plotter: Plotter,
                transparent: bool,
                space: TimespaceDomain,
                diffusion: Callable
            ):
                self.file_prefix = file_prefix
                self.title = title
                self.plotter = plotter
                self.transparent = transparent
                self.space = space
                self.diffusion = diffusion
                self.i = 0
                self.frame_file_names = []
                self.D = None

            def __call__(self, t, u):
                assert u.shape[0] == u.shape[1]
                spatial_resolution = u.shape[0]
                frameName = f"{self.file_prefix}_{self.i}.png"
                points = self.plotter.dpi * \
                    max(self.plotter.figsize[0], self.plotter.figsize[1])
                space = UniformSpace(self.space, spatial_resolution, 0)
                x, y, _ = space.getInitialPointsKeepDims()
                if spatial_resolution > 2 * points:
                    indices = list(range(0, spatial_resolution,
                                   spatial_resolution//points + 1))
                    x = x[indices][:, indices]
                    y = y[indices][:, indices]
                    u = u[indices][:, indices]
                x = x.squeeze(axis=-1)
                y = y.squeeze(axis=-1)
                title = make_title(self.title + [f"(t={t:4.2f})"])
                if self.diffusion is None:
                    _ = self.plotter.plot(
                        u,
                        x,
                        y,
                        title,
                    )
                else:
                    if self.D is None:
                        D = self.diffusion(x, y)
                    _ = self.plotter.plotWithBackground(
                        u,
                        D,
                        x,
                        y,
                        title,
                    )
                self.frame_file_names.append(frameName)
                plt.savefig(
                    frameName,
                    transparent=self.transparent,
                    facecolor="white",
                )
                self.i += 1

        iteration = Iteration(
            self.file_prefix,
            self.title,
            plotter,
            self.transparent,
            data_provider.timespace_domain,
            diffusion
        )

        data_provider.for_each_frame(iteration)
        self.__makeGif(frameFileNames=iteration.frame_file_names)

    def __makeGif(self, frameFileNames):
        try:
            from PIL import Image

            frames = []
            for file in frameFileNames:
                image = Image.open(file)
                frames.append(image)
            frames[0].save(
                f"{self.file_prefix}.gif",
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
