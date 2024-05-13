class Grid:
    def __init__(self, experiment, points_num, device):
        import torch
        x_domain = experiment.timespaceDomain.spaceDomains[0]
        x_start = x_domain[0]
        x_end = x_domain[1]
        x_linspace = torch.linspace(x_start, x_end, points_num)

        y_domain = experiment.timespaceDomain.spaceDomains[1]
        y_start = y_domain[0]
        y_end = y_domain[1]
        y_linspace = torch.linspace(y_start, y_end, points_num)

        self.x, self.y = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
        self.x = self.x.to(device)
        self.y = self.y.to(device)

        self.dx = (x_end - x_start) / (points_num - 1)
        self.sqdx = self.dx**2
        self.dy = (y_end - y_start) / (points_num - 1)
        self.sqdy = self.dy**2
