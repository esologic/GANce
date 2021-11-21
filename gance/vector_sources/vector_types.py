"""
Common types to describe different collections of data.

Technically, since we're describing vectors and matrices here, you could call them Tensors.
But too much stuff is called vector to rename things. TODO -- change language to tensors.

I'd LOVE to be able to use numpy typing to describe the shapes of these things per:
https://numpy.org/devdocs/reference/typing.html
However, we're hard locked to `numpy==1.16.4` to stay compatible with stylegan.

Pylint does not mix well with NewTypes. So, in the `.pylintrc` for this project, there
are manual overrides to ignore `no-member` errors for this class.
"""

from typing import NamedTuple, NewType, Union

import numpy as np  # pylint: disable=unused-import

# dimensions are (Any,)
SingleVector = NewType("SingleVector", "np.ndarray[np.float32]")  # type: ignore

# dimensions are (Any,)
ConcatenatedVectors = NewType("ConcatenatedVectors", "np.ndarray[np.float32]")  # type: ignore

# dimensions are (Any, Any)
DividedVectors = NewType("DividedVectors", "np.ndarray[np.float32]")  # type: ignore

# dimensions are (Any, Any)
SingleMatrix = NewType("SingleMatrix", "np.ndarray[np.float32]")  # type: ignore

# dimensions are (Any, Any)
ConcatenatedMatrices = NewType("ConcatenatedMatrices", "np.ndarray[np.float32]")  # type: ignore

# dimensions are (Any, Any, Any)
DividedMatrices = NewType("DividedMatrices", "np.ndarray[np.float32]")  # type: ignore


class VectorsLabel(NamedTuple):
    """
    Intermediate type, used to link a vector array to the name of it for visualization.
    """

    data: ConcatenatedVectors  # PRetty much everywhere this means concatenated version
    vector_length: int
    label: str


class MatricesLabel(NamedTuple):
    """
    Intermediate type, used to link a vector array to the name of it for visualization.
    """

    data: ConcatenatedMatrices
    vector_length: int
    label: str


def is_vector(data: Union[SingleVector, SingleMatrix, np.ndarray]) -> bool:
    """
    Check the shape of a given array to figure out if it's a vector or not.
    :param data: Array to evaluate.
    :return: If the input is a vector. If False, we can assume it's a Matrix or something bigger.
    """

    if len(data.shape) < 2:
        return True

    return False
