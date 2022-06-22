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

from typing import Any, NamedTuple, NewType, Union

import numpy as np  # pylint: disable=unused-import
import numpy.typing as npt

# dimensions are (Any,)
SingleVector = NewType("SingleVector", npt.NDArray[np.float32])

# dimensions are (Any,)
ConcatenatedVectors = NewType("ConcatenatedVectors", npt.NDArray[np.float32])

# dimensions are (Any, Any)
DividedVectors = NewType("DividedVectors", npt.NDArray[np.float32])

# dimensions are (Any, Any)
SingleMatrix = NewType("SingleMatrix", npt.NDArray[np.float32])

# dimensions are (Any, Any)
ConcatenatedMatrices = NewType("ConcatenatedMatrices", npt.NDArray[np.float32])

# dimensions are (Any, Any, Any)
DividedMatrices = NewType("DividedMatrices", npt.NDArray[np.float32])


class VectorsLabel(NamedTuple):
    """
    Intermediate type, used to link a vector array to the name of it for visualization.
    """

    data: ConcatenatedVectors  # Pretty much everywhere this means concatenated version
    vector_length: int
    label: str


class MatricesLabel(NamedTuple):
    """
    Intermediate type, used to link a vector array to the name of it for visualization.
    """

    data: ConcatenatedMatrices
    vector_length: int
    label: str


def is_vector(data: Union[SingleVector, SingleMatrix, npt.NDArray[Any]]) -> bool:
    """
    Check the shape of a given array to figure out if it's a vector or not.
    :param data: Array to evaluate.
    :return: If the input is a vector. If False, we can assume it's a Matrix or something bigger.
    """

    if len(data.shape) < 2:
        return True

    return False
