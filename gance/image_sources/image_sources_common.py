from typing import NamedTuple

from gance.gance_types import RGBInt8ImageType


class ImageResolution(NamedTuple):

    width: int
    height: int


def image_resolution(image: RGBInt8ImageType) -> ImageResolution:
    """

    :param image:
    :return:
    """

    return ImageResolution(height=image.shape[0], width=image.shape[1])