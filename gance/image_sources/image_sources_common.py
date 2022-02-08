"""
Common functionality and types used in still images and in video.
"""

from typing import NamedTuple

from gance.gance_types import RGBInt8ImageType


class ImageResolution(NamedTuple):
    """
    Standard NT for image dimensions. Creators are responsible for making sure the order is
    correct.
    """

    width: int
    height: int


def image_resolution(image: RGBInt8ImageType) -> ImageResolution:
    """
    Get an image's resolution.
    :param image: To size.
    :return: Image resolution as an NT.
    """

    return ImageResolution(height=image.shape[0], width=image.shape[1])
