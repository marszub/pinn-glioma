from typing import Tuple
from torch import Tensor, sqrt


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
