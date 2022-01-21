"""
Functionality for working with still images.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from gance.gance_types import RGBInt8ImageType

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


def read_image(image_path: Path) -> RGBInt8ImageType:
    """
    Read an image from disk into the canonical, in-memory format.
    :param image_path: Path to the image file on disk.
    :return: The image
    """
    # Verified by hand that this cast is valid
    return RGBInt8ImageType(np.asarray(Image.open(str(image_path))))
