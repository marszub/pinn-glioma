import torch
from TimespaceDomain import TimespaceDomain
from simulationSpace.SampleSpace import SampleSpace

class UniformSpace(SampleSpace):
    def __init__(self, timespaceDomain: TimespaceDomain, spaceResoultion: int, timeResoultion: int, requiresGrad: bool = True):
        super().__init__(timespaceDomain)
        self.spaceResoultion = spaceResoultion
        self.timeResoultion = timeResoultion
        self.requiresGrad = requiresGrad
    
    def __getInitialPointsNoGrad(self):
        x_domain = self.timespaceDomain.spaceDomains[0]
        y_domain = self.timespaceDomain.spaceDomains[1]
        t_domain = self.timespaceDomain.timeDomain
        x_linspace = torch.linspace(x_domain[0], x_domain[1], self.spaceResoultion)
        y_linspace = torch.linspace(y_domain[0], y_domain[1], self.spaceResoultion)
        x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
        x_grid = x_grid.unsqueeze(dim=-1).to(self.device)
        y_grid = y_grid.unsqueeze(dim=-1).to(self.device)
        t0 = torch.full_like(x_grid, t_domain[0])
        return (x_grid, y_grid, t0)

    def getInitialPointsKeepDims(self):
        """Keeps 2D"""
        x_grid, y_grid, t0 = self.__getInitialPointsNoGrad()
        x_grid.requires_grad = self.requiresGrad
        y_grid.requires_grad = self.requiresGrad
        t0.requires_grad = self.requiresGrad
        return (x_grid, y_grid, t0)


    def getInitialPoints(self):
        """Generates tensor of points convering initial condition"""
        x_grid, y_grid, t0 = self.__getInitialPointsNoGrad()
        x_grid = x_grid.reshape(-1, 1)
        y_grid = y_grid.reshape(-1, 1)
        t0 = t0.reshape(-1, 1)
        x_grid.requires_grad = self.requiresGrad
        y_grid.requires_grad = self.requiresGrad
        t0.requires_grad = self.requiresGrad
        return (x_grid, y_grid, t0)
        

    def getInteriorPointsKeepDims(self):
        """Keeps 2D"""
        x_domain = self.timespaceDomain.spaceDomains[0]
        y_domain = self.timespaceDomain.spaceDomains[1]
        t_domain = self.timespaceDomain.timeDomain
        assert(self.spaceResoultion == self.timeResoultion)
        n_points = self.spaceResoultion
        x_raw = torch.linspace(
            x_domain[0],
            x_domain[1],
            steps=n_points,
            requires_grad=self.requiresGrad,
        )
        y_raw = torch.linspace(
            y_domain[0],
            y_domain[1],
            steps=n_points,
            requires_grad=self.requiresGrad,
        )
        t_raw = torch.linspace(
            t_domain[0],
            t_domain[1],
            steps=n_points,
            requires_grad=self.requiresGrad,
        )
        grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

        x = grids[0].to(self.device)
        y = grids[1].to(self.device)
        t = grids[2].to(self.device)
        return x, y, t
    
    def getInteriorPoints(self):
        """Generates tensor of points convering interior of simulation"""
        x, y, t = self.getInteriorPointsKeepDims()
        return x.reshape(-1, 1), y.reshape(-1, 1), t.reshape(-1, 1)

    def getBoundaryPoints(self):
        """
        Generates tensor of points convering space boundary
             .+------+
           .' |    .'|
          +---+--+'  |
          |   |  |   |
        y |  ,+--+---+
          |.'    | .' t
          +------+'
              x
        """
        x_domain = self.timespaceDomain.spaceDomains[0]
        y_domain = self.timespaceDomain.spaceDomains[1]
        t_domain = self.timespaceDomain.timeDomain
        assert(self.spaceResoultion == self.timeResoultion)
        n_points = self.spaceResoultion
        x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
        y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
        t_linspace = torch.linspace(t_domain[0], t_domain[1], n_points)

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _ = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = x_grid.reshape(-1, 1).to(self.device)
        x_grid.requires_grad = self.requiresGrad
        y_grid = y_grid.reshape(-1, 1).to(self.device)
        y_grid.requires_grad = self.requiresGrad
        t_grid = t_grid.reshape(-1, 1).to(self.device)
        t_grid.requires_grad = self.requiresGrad

        x0 = torch.full_like(t_grid, x_domain[0], requires_grad=self.requiresGrad)
        x1 = torch.full_like(t_grid, x_domain[1], requires_grad=self.requiresGrad)
        y0 = torch.full_like(t_grid, y_domain[0], requires_grad=self.requiresGrad)
        y1 = torch.full_like(t_grid, y_domain[1], requires_grad=self.requiresGrad)

        down = (x_grid, y0, t_grid)
        up = (x_grid, y1, t_grid)
        left = (x0, y_grid, t_grid)
        right = (x1, y_grid, t_grid)

        return down, up, left, right
