#!/bin/python

from model.Experiment import Experiment
from simulate.ArgsParser import ArgsParser
from simulate.Initializer import Initializer
import torch


if __name__ == "__main__":
    argsParser = ArgsParser()
    argsParser.show()
    args = argsParser.get()
    experiment = Experiment()
    initializer = Initializer(args, experiment)
    saver = initializer.get_saver()
    grid = initializer.get_grid()

    u = initializer.get_initial_state()
    rho = experiment.rho
    x, y = grid.x[1:-1, 1:-1], grid.y[1:-1, 1:-1]
    sqdx, sqdy = grid.sqdx, grid.sqdy
    dt = initializer.get_dt()
    D = experiment.diffusion(x, y)

    saver.first_save(u, torch.tensor(0.0))
    for t in initializer.get_iterator():
        R = experiment.treatment(x, y, t)

        udx = u[:, 1:-1]
        dudx2 = (udx[:-2, :] - 2*udx[1:-1, :] + udx[2:, :]) / sqdx
        udy = u[1:-1, :]
        dudy2 = (udy[:, :-2] - 2*udy[:, 1:-1] + udy[:, 2:]) / sqdy
        uf = u[1:-1, 1:-1]

        rhs = D * dudx2 + D * dudy2 + rho * uf * (1 - uf) - R * uf

        u[1:-1, 1:-1] = uf + rhs * dt

        saver.mid_save(u, t)
    saver.last_save(u, t)
