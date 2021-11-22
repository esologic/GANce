"""
Common types used across this project
"""
from pathlib import Path
from typing import NamedTuple, NewType, Optional, Tuple

import numpy as np  # pylint: disable=unused-import


class LabeledCoordinates(NamedTuple):
    """
    Label the coordinates of the bounding box
    """

    top: int
    right: int
    bottom: int
    left: int


class PathAndBoundingBoxes(NamedTuple):
    """
    Map the path to an image to the face bounding boxes within that image
    """

    path_to_image: Path
    bounding_boxes: Optional[Tuple[LabeledCoordinates, ...]]


# dimensions are (Width, Height, Colors)
RGBInt8ImageType = NewType("RGBInt8ImageType", "np.ndarray[np.uint8]")  # type: ignore
