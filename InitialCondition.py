from typing import Tuple
from torch import Tensor, sqrt
from TimespaceDomain import TimespaceDomain

def initial_condition(x: Tensor, y: Tensor) -> Tensor:
    d = sqrt((x - 0.6) ** 2 + (y - 0.6) ** 2)
    res = -(d**2) - 4 * d + 0.4
    res = res * (res > 0)
    # res = torch.exp(-(d*7)**2) / 2
    return res


def initial_condition_new(
    x: Tensor, y: Tensor
) -> Tensor:
    r = sqrt((x - 0.6) ** 2 + (y - 0.6) ** 2)
    return -20 * (r - 0.05) * (r + 0.05) * (r < 0.05)

class InitialCondition:
    def __init__(self, origin: Tuple[float, float], h: float, r: float):
        self.origin = origin
        self.a = -h/(2*r**2)
        self.b = -h/(2*r)
        self.c = h

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        xOrigin, yOrigin = self.origin
        d = sqrt((x - xOrigin) ** 2 + (y - yOrigin) ** 2)
        res = self.a * d**2 + self.b * d + self.c
        res = res * (res > 0)
        return res