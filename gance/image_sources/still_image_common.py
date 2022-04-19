"""
Functionality for working with still images.
"""

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from PIL import Image

from gance.gance_types import RGBInt8ImageType
from gance.image_sources import image_sources_common
from gance.logger_common import LOGGER

PNG = "png"


def write_image(image: RGBInt8ImageType, path: Path) -> None:
    """
    Writes a given image to the path.
    Uses PNG by default.
    :param image: Image in memory.
    :param path: Destination.
    :return: None
    """

    Image.fromarray(image).save(fp=str(path), format=PNG.upper())


def read_image(image_path: Path, mode: Optional[str] = None) -> RGBInt8ImageType:
    """
    Read an image from disk into the canonical, in-memory format.
    :param image_path: Path to the image file on disk.
    :param mode: If given, image will be converted with PIL to this mode.
    :return: The image
    """

    im = Image.open(str(image_path))

    if mode is not None:
        im = im.convert(mode)

    # Verified by hand that this cast is valid
    return RGBInt8ImageType(np.asarray(im))


def horizontal_concat_images(images: Iterator[RGBInt8ImageType]) -> RGBInt8ImageType:
    """
    Helper function. Adds logging.
    :param images: To concatenate.
    :return: Concatenated image.
    """
    images_as_list = list(images)
    LOGGER.debug(
        f"Horizontally concatenating {len(images_as_list)} images, "
        f"sizes: {[image_sources_common.image_resolution(image) for image in images_as_list]}"
    )
    output: RGBInt8ImageType = cv2.hconcat(images_as_list)
    return output
