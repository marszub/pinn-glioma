from matplotlib import pyplot as plt, colors
import numpy as np
from torch import Tensor


class Plotter:
    def __init__(
        self,
        figsize=(5, 6),
        dpi=200,
        cmap="viridis",
        limit=None
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.limit = limit
        self.levels = 10
        cmap_obj = plt.get_cmap(cmap)
        rgb = cmap_obj(
            np.arange(0, cmap_obj.N, cmap_obj.N//self.levels))[:, :3]
        alpha = 1. - np.min(rgb, axis=1)
        alpha = np.expand_dims(alpha, -1)
        rgba = np.concatenate(((rgb + alpha - 1) / alpha, alpha), axis=1)
        rgba[0] = np.array([0, 0, 0, 0])

        self.front_cmap = colors.ListedColormap(
            rgba, name=cmap + "_alpha")
        self.cmap = colors.ListedColormap(
            rgb, name=cmap + "_steps")

    def plot(
        self,
        z: Tensor,
        x: Tensor,
        y: Tensor,
        title,
    ):
        fig, ax = plt.subplots(figsize=self.figsize,
                               dpi=self.dpi, num=1, clear=True)
        X = x.detach().cpu().numpy().squeeze()
        Y = y.detach().cpu().numpy().squeeze()
        Z = z.detach().cpu().numpy().squeeze()
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        c = ax.pcolormesh(X, Y, Z, cmap=self.cmap)
        if self.limit is not None:
            c.set_clim(0, self.limit)
        fig.colorbar(c, ax=ax)

        return fig

    def plotWithBackground(
        self,
        z: Tensor,
        backgroundZ: Tensor,
        x: Tensor,
        y: Tensor,
        title,
    ):
        fig, ax = plt.subplots(figsize=self.figsize,
                               dpi=self.dpi, num=1, clear=True)
        X = x.detach().cpu().numpy()
        Y = y.detach().cpu().numpy()
        Z = z.detach().cpu().numpy()
        backgroundZ = backgroundZ.detach().cpu().numpy()
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        ax.pcolormesh(
            X.squeeze(),
            Y.squeeze(),
            backgroundZ.squeeze(),
            cmap="Greys",
            vmin=0,
            vmax=2*np.max(backgroundZ),
        )
        lim = self.limit if self.limit is not None else Z.max()
        c = ax.pcolormesh(
            X.squeeze(),
            Y.squeeze(),
            Z.squeeze(),
            cmap=self.front_cmap,
        )
        if self.limit is not None:
            c.set_clim(0, self.limit)
        fig.colorbar(c, ax=ax)

        return fig
