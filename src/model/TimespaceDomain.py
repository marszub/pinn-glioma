from typing import List, NamedTuple, Tuple


class TimespaceDomain(NamedTuple):
    DimDomain = Tuple[float, float]
    spaceDomains: List[DimDomain]
    timeDomain: DimDomain

    def get_points_per_space_unit(self, points_num):
        xSpaceSize = (
            self.spaceDomains[0][1]
            - self.spaceDomains[0][0]
        )
        ySpaceSize = (
            self.spaceDomains[1][1]
            - self.spaceDomains[1][0]
        )
        return points_num / (xSpaceSize * ySpaceSize)
