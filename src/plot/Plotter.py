from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor


class Plotter:
    def __init__(
        self,
        figsize=(8, 6),
        dpi=100,
        cmap="viridis",
        limit=None
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap
        self.limit = limit

    def plot(
        self,
        z: Tensor,
        x: Tensor,
        y: Tensor,
        title,
    ):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        X = x.detach().cpu().numpy()
        Y = y.detach().cpu().numpy()
        Z = z.detach().cpu().numpy()
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        c = ax.pcolormesh(
            np.squeeze(X, axis=-1),
            np.squeeze(Y, axis=-1),
            np.squeeze(Z, axis=-1),
            cmap=self.cmap,
        )
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
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        X = x.detach().cpu().numpy()
        Y = y.detach().cpu().numpy()
        Z = z.detach().cpu().numpy()
        backgroundZ = backgroundZ.detach().cpu().numpy()
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        ax.pcolormesh(
            np.squeeze(X, axis=-1),
            np.squeeze(Y, axis=-1),
            np.squeeze(backgroundZ, axis=-1),
            cmap="Greys",
            vmin=0,
            vmax=2*np.max(backgroundZ),
        )
        lim = self.limit if self.limit is not None else Z.max()
        alpha = np.minimum((Z-Z.min())/(lim-Z.min()), 1.0)
        c = ax.pcolormesh(
            np.squeeze(X, axis=-1),
            np.squeeze(Y, axis=-1),
            np.squeeze(Z, axis=-1),
            cmap=self.cmap,
            alpha=alpha,
        )
        if self.limit is not None:
            c.set_clim(0, self.limit)
        fig.colorbar(c, ax=ax)

        return fig