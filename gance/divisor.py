import math
from typing import Union


def divide_no_remainder(numerator: Union[int, float], denominator: Union[int, float]) -> int:

    frac, repeats = math.modf(numerator / denominator)

    if frac != 0:
        raise ValueError(f"Cannot evenly divide {numerator} into {denominator}")

    return int(repeats)
