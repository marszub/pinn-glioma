from typing import NamedTuple

class Weights(NamedTuple):
    residual: float
    initial: float
    boundary: float

    def normalized(self):
        total = self.residual + self.initial + self.boundary
        return Weights(
            residual=self.residual / total,
            initial=self.initial / total,
            boundary=self.boundary / total,
        )
