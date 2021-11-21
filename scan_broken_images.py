"""
Utilities to make sure training images are valid.
"""

import logging
import os
import pprint
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import click
import PIL
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)20s - %(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


def _open_image(path_to_image: Path) -> Optional[Path]:
    """
    Return the path to the image if it is broken, return None otherwise.
    :param path_to_image: The image to check.
    :return: Return the path to the image if it is broken, return None otherwise.
    """
    try:
        img = Image.open(path_to_image)
        img.load()
    except (SyntaxError, PIL.UnidentifiedImageError, OSError) as e:  # pylint: disable=no-member
        LOGGER.warning(f"Found a broken image: {path_to_image}. Error: {pprint.pformat(e)}")
        return path_to_image

    return None


@click.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help=(
        "Opens each image in the given directory with PIL to see if it can be done without error. "
        "If the image is broken, meaning it cannot be opened, it is deleted."
    ),
)
@click.option(
    "--no_delete",
    is_flag=True,
    help="If given, the files will not be deleted only listed.",
    default=False,
    show_default=True,
)
def scan_for_broken_images(directory: str, no_delete: bool) -> None:
    """
    Scan a given directory for images that cannot be opened by pillow.
    Delete any of these problem images and log that you've done so.

    \f

    :param directory: Path to directory with images to scan.
    :param no_delete: If given, the files will not be deleted only listed.
    :return: None
    """

    LOGGER.info(f"Scanning {directory} for broken images...")

    image_paths = Path(directory).glob("*.jpeg")

    with Pool() as p:
        scanned_images = p.map(func=_open_image, iterable=image_paths)

    broken_image_paths = list(filter(None, scanned_images))

    LOGGER.info(f"Found: {len(broken_image_paths)} broken images.")

    if not no_delete:
        LOGGER.info("Deleting them now...")
        for path in broken_image_paths:
            path_as_string = str(path)
            LOGGER.info(f"Deleting: {path_as_string}")
            os.remove(path_as_string)


if __name__ == "__main__":
    scan_for_broken_images()  # pylint: disable=no-value-for-parameter
