import torch
from SampleSpace import SampleSpace

def get_boundary_points(
    space: SampleSpace,
    device=torch.device("cpu"),
    requires_grad=True,
):
    """
         .+------+
       .' |    .'|
      +---+--+'  |
      |   |  |   |
    y |  ,+--+---+
      |.'    | .' t
      +------+'
          x
    """

    x_domain = space.timespaceDomain.spaceDomains[0]
    y_domain = space.timespaceDomain.spaceDomains[1]
    t_domain = space.timespaceDomain.timeDomain
    assert(space.spaceResoultion == space.timeResoultion)
    n_points = space.spaceResoultion
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    t_linspace = torch.linspace(t_domain[0], t_domain[1], n_points)

    x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
    y_grid, _ = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t_grid = t_grid.reshape(-1, 1).to(device)
    t_grid.requires_grad = requires_grad

    x0 = torch.full_like(t_grid, x_domain[0], requires_grad=requires_grad)
    x1 = torch.full_like(t_grid, x_domain[1], requires_grad=requires_grad)
    y0 = torch.full_like(t_grid, y_domain[0], requires_grad=requires_grad)
    y1 = torch.full_like(t_grid, y_domain[1], requires_grad=requires_grad)

    down = (x_grid, y0, t_grid)
    up = (x_grid, y1, t_grid)
    left = (x0, y_grid, t_grid)
    right = (x1, y_grid, t_grid)

    return down, up, left, right


def get_initial_points(
    space: SampleSpace,
    device=torch.device("cpu"),
    requires_grad=True,
):
    x_domain = space.timespaceDomain.spaceDomains[0]
    y_domain = space.timespaceDomain.spaceDomains[1]
    t_domain = space.timespaceDomain.timeDomain
    x_linspace = torch.linspace(x_domain[0], x_domain[1], space.spaceResoultion)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], space.spaceResoultion)
    x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)


def get_interior_points(
    space: SampleSpace,
    device=torch.device("cpu"),
    requires_grad=True,
):
    x_domain = space.timespaceDomain.spaceDomains[0]
    y_domain = space.timespaceDomain.spaceDomains[1]
    t_domain = space.timespaceDomain.timeDomain
    assert(space.spaceResoultion == space.timeResoultion)
    n_points = space.spaceResoultion
    x_raw = torch.linspace(
        x_domain[0],
        x_domain[1],
        steps=n_points,
        requires_grad=requires_grad,
    )
    y_raw = torch.linspace(
        y_domain[0],
        y_domain[1],
        steps=n_points,
        requires_grad=requires_grad,
    )
    t_raw = torch.linspace(
        t_domain[0],
        t_domain[1],
        steps=n_points,
        requires_grad=requires_grad,
    )
    grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

    x = grids[0].reshape(-1, 1).to(device)
    y = grids[1].reshape(-1, 1).to(device)
    t = grids[2].reshape(-1, 1).to(device)
    return x, y, t