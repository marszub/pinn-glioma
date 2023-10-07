from typing import List, NamedTuple, Tuple

class TimespaceDomain(NamedTuple):
    DimDomain = Tuple[float, float]
    spaceDomains: List[DimDomain]
    timeDomain: DimDomain
