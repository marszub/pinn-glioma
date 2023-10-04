from matplotlib import pyplot as plt
from torch import Tensor

from plotter.Plotter import Plotter


class Plotter3D(Plotter):
    def __init__(
        self,
        figsize=(8, 6),
        limit=0.2,
    ):
        self.figsize = figsize
        self.limit = limit

    def plot(
        self,
        z: Tensor,
        x: Tensor,
        y: Tensor,
        nPointsX,
        nPointsY,
        title,
    ):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(projection="3d")
        z_raw = z.detach().cpu().numpy()
        x_raw = x.detach().cpu().numpy()
        y_raw = y.detach().cpu().numpy()
        X = x_raw.reshape(nPointsX, nPointsY)
        Y = y_raw.reshape(nPointsX, nPointsY)
        Z = z_raw.reshape(nPointsX, nPointsY)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if self.limit is not None:
            ax.axes.set_zlim3d(bottom=0, top=self.limit)
        c = ax.plot_surface(X, Y, Z)
        return fig
