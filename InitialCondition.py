from torch import Tensor, sqrt

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