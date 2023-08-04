from typing import Callable
import numpy as np
from Pinn import PINN
import matplotlib.pyplot as plt
from SampleSpace import SampleSpace
from Samples import get_initial_points
from torch import Tensor, full_like, unique, ones_like
from matplotlib.animation import FuncAnimation


def running_average(y, window=100):
    cumsum = np.cumsum(y, axis=0)
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def plot_color(
    z: Tensor,
    x: Tensor,
    y: Tensor,
    n_points_x,
    n_points_t,
    title,
    figsize=(8, 6),
    dpi=100,
    cmap="viridis",
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    c = ax.pcolormesh(X, Y, Z, cmap=cmap)
    fig.colorbar(c, ax=ax)

    return fig


def plot_3D(
    z: Tensor,
    x: Tensor,
    y: Tensor,
    n_points_x,
    n_points_t,
    title,
    figsize=(8, 6),
    dpi=100,
    limit=0.2,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    z_raw = z.detach().cpu().numpy()
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()
    X = x_raw.reshape(n_points_x, n_points_t)
    Y = y_raw.reshape(n_points_x, n_points_t)
    Z = z_raw.reshape(n_points_x, n_points_t)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axes.set_zlim3d(bottom=-limit, top=limit)
    c = ax.plot_surface(X, Y, Z)
    return fig


def plot_solution(
    pinn: PINN, x: Tensor, t: Tensor, figsize=(8, 6), dpi=100
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_raw = unique(x).reshape(-1, 1)
    t_raw = unique(t)

    def animate(i):
        if not i % 10 == 0:
            t_partial = ones_like(x_raw) * t_raw[i]
            f_final = f(pinn, x_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy(),
                f_final.detach().numpy(),
                label=f"Time {float(t[i])}",
            )
            ax.set_ylim(-1, 1)
            ax.legend()

    n_frames = t_raw.shape[0]
    return FuncAnimation(
        fig, animate, frames=n_frames, interval=100, repeat=False
    )


def write_loss(losses):
    print(f"Total loss: \t{losses[0]:.5f} ({losses[0]:.3E})")
    print(f"Interior loss: \t{losses[1]:.5f} ({losses[1]:.3E})")
    print(f"Initial loss: \t{losses[2]:.5f} ({losses[2]:.3E})")
    print(f"Bondary loss: \t{losses[3]:.5f} ({losses[3]:.3E})")


def plot_losses(loss_over_time, filePath, labels=[]):
    average_loss = running_average(loss_over_time, window=100)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_title("Loss function (runnig average)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(average_loss, label=labels)
    ax.set_yscale("log")
    if len(labels) > 0:
        plt.legend()
    plt.savefig(filePath)
    plt.close()


def plot_initial_condition(space: SampleSpace, initialCondition: Callable, filePath):
    x, y, _ = get_initial_points(space, requires_grad=False)
    z = initialCondition(x, y)
    fig = plot_color(
        z,
        x,
        y,
        space.spaceResoultion,
        space.spaceResoultion,
        "Initial condition - exact",
    )
    plt.savefig(filePath, transparent=True)
    plt.close()


def animate_progress(pinn: PINN, space: SampleSpace, dirPath):
    for i in range(space.timeResoultion):
        x, y, _ = get_initial_points(space, requires_grad=False)
        t = full_like(
            x,
            space.timespaceDomain.timeDomain[0]
            + i
            * (
                space.timespaceDomain.timeDomain[1]
                - space.timespaceDomain.timeDomain[0]
            )
            / space.timeResoultion,
        )
        z = pinn(x, y, t)
        fig = plot_color(
            z, x, y, space.spaceResoultion, space.spaceResoultion, "PINN"
        )
        plt.savefig(
            dirPath + f"/img_{i}.png", transparent=True, facecolor="white"
        )
        plt.close()
