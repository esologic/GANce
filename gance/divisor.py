"""
Canonical function that needs to work the same across multiple use cases.
Silly to have its own file but here we are.
"""

import math
from typing import Union


def divide_no_remainder(numerator: Union[int, float], denominator: Union[int, float]) -> int:
    """
    Raise a Value Error if the division is not even.
    Return `numerator` / `denominator`.
    :param numerator: See top.
    :param denominator: See top.
    :return: `numerator` / `denominator`.
    """

    frac, repeats = math.modf(numerator / denominator)

    if frac != 0:
        raise ValueError(f"Cannot evenly divide {numerator} into {denominator}")

    return int(repeats)
