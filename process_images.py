"""
Tools to prepare images for styleGAN training.
"""

import json
import logging
import multiprocessing
import os
import pprint
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

import click
import cv2
import PIL
from PIL import Image

from gance.logger_common import LOGGER
from gance.select_good_face_images import SourceDestination, copy, select_images_for_training

DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)20s - %(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """
    Tools to prepare images for styleGAN training.

    \f

    :return: None
    """


def scale_image(
    output_directory: str, scaled_image_width: int, scaled_image_height: int, image_path: Path
) -> None:
    """
    Scale the given image to the target size, write the resulting scaled image to the standard
    folder containing scaled images.
    :param output_directory: The directory the scaled images will be placed in.
    :param scaled_image_width: The width of the scaled image.
    :param scaled_image_height: The height of the scaled image.
    :param image_path: The path to the image to scale
    :return: None
    """
    original_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(original_image, (scaled_image_width, scaled_image_height))
    cv2.imwrite(
        str(
            Path(output_directory).joinpath(
                f"{scaled_image_width}_{scaled_image_height}_{image_path.name}"
            )
        ),
        resized,
    )


@cli.command()
@click.option(
    "--original-images-directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help="A directory containing the images to scale.",
)
@click.option(
    "--image-file-extension",
    type=str,
    default="jpeg",
    help="The file extension for the images in original_images_directory.",
    show_default=True,
)
@click.option(
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help="The directory the scaled images will be placed in.",
)
@click.option(
    "--scaled-image-width",
    type=click.IntRange(min=0),
    default=DEFAULT_IMAGE_WIDTH,
    help="The width of the scaled image.",
    show_default=True,
)
@click.option(
    "--scaled-image-height",
    type=click.IntRange(min=0),
    default=DEFAULT_IMAGE_HEIGHT,
    help="The height of the scaled image.",
    show_default=True,
)
def scale_images(
    original_images_directory: str,
    image_file_extension: str,
    output_directory: str,
    scaled_image_width: int,
    scaled_image_height: int,
) -> None:
    """
    Given the cropped images, scale them to sizes usable by styleGAN. Scale images using multiple
    processes.
    :param original_images_directory: A directory containing the images to scale.
    :param image_file_extension: The file extension for the images in original_images_directory.
    :param output_directory: The directory the scaled images will be placed in.
    :param scaled_image_width: The width of the scaled image.
    :param scaled_image_height: The height of the scaled image.
    :return: None
    """

    cropped_face_images_paths = Path(original_images_directory).glob(f"*.{image_file_extension}")

    with multiprocessing.Pool() as p:
        p.map(
            partial(scale_image, output_directory, scaled_image_width, scaled_image_height),
            cropped_face_images_paths,
        )


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


@cli.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help=(
        "Opens each image in the given directory with PIL to see if it can be done without error. "
        "If the image is broken, meaning it cannot be opened, it is deleted."
    ),
)
@click.option(
    "--no-delete",
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


@cli.command()
@click.option(
    "--primary-directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help=(
        "Directory of images to search. Images that contain faces are highest priority and will be "
        "selected over all other images in other directories."
    ),
    multiple=True,
    required=True,
)
@click.option(
    "--secondary-directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help=(
        "Directory of images to search. Images that contain faces will be selected if there aren't "
        "enough faces in primary directories to reach the value given by --target_num_images."
    ),
    multiple=True,
)
@click.option(
    "--target-num-images",
    type=click.IntRange(min=1),
    help=(
        "The desired number of images to select. If this number cannot be reached using face "
        "images, images near (in time) face images will be selected. If even this is not "
        "enough, images are randomly selected until the number is met. If there aren't enough "
        "images, all images in all directories will be selected."
    ),
    default=None,
)
@click.option(
    "--destination-directory",
    type=click.Path(exists=False, file_okay=False, readable=True, dir_okay=True),
    help="Selected images are copied to this directory.",
    required=False,
    default=None,
)
@click.option(
    "--summary",
    type=click.Path(exists=False, file_okay=True, readable=True, dir_okay=False),
    help="A summary about the selected images is written to this file.",
    required=False,
    default=None,
)
def select_images_copy(
    primary_directory: List[str],
    secondary_directory: List[str],
    target_num_images: Optional[int],
    destination_directory: Optional[str],
    summary: Optional[str],
) -> Optional[int]:
    """
    CLI to select images for training. User provides the target number of images to select, and
    primary/secondary directories full of images to select from.

    To reach the target number of images, images are selected in this order:

    1. Images that contain faces in primary directories
    2. Images that contain faces in secondary directories
    3. Images that were captured 2 before or two after images with faces in them in time
    4. Randomly chosen, unselected images (no duplicates)

    Images that cannot be opened with PIL (corrupted images) will not be selected.

    \f # Truncate docs for click

    :param primary_directory: Directory of images to search. Images that contain faces are highest
    priority and will be selected over all other images in other directories.
    :param secondary_directory: Directory of images to search. Images that contain faces will be
    selected if there aren't enough faces in primary directories to reach the value given
    by `target_num_images`.
    :param target_num_images: The desired number of images to select. If this number cannot be
    reached using face images, images near (in time) face images will be selected. If even this is
    not enough, images are randomly selected until the number is met. If there aren't enough images,
    all images in all directories will be selected.
    :param destination_directory: Selected images are copied to this directory.
    :param summary: A summary about the selected images is written to this file.
    :return:
    """

    LOGGER.info("Detecting faces...")

    to_copy = select_images_for_training(primary_directory, secondary_directory, target_num_images)

    LOGGER.info("Face detection finished. Results:")

    LOGGER.info(
        "* Number of images with faces in primary/secondary directories: "
        f"{to_copy.num_img_w_faces}"
    )
    LOGGER.info(
        "* Number of images surrounding images with faces in primary/secondary directories: "
        f"{to_copy.num_img_around_img_w_faces}"
    )
    LOGGER.info(
        "* Number of images other images, not in first two sets: "
        f"{to_copy.num_randomized_img_wout_faces}"
    )

    total_num_images = (
        to_copy.num_img_w_faces
        + to_copy.num_img_around_img_w_faces
        + to_copy.num_randomized_img_wout_faces
    )

    LOGGER.info(f"Total number of images: {total_num_images}")

    if summary is not None:

        with open(summary, "w") as fp:
            json.dump(
                {
                    "total_images": total_num_images,
                    "num_images_with_faces": to_copy.num_img_w_faces,
                    "num_randomized_images_without_faces": to_copy.num_randomized_img_wout_faces,
                    "num_images_surrounding_images_with_faces": to_copy.num_img_around_img_w_faces,
                },
                fp,
            )

    if destination_directory is not None:

        destination = Path(destination_directory)
        destination.mkdir(exist_ok=True)

        copy_paths = to_copy.path_and_bounding_boxes

        with Pool() as p:
            p.map(
                copy,
                [
                    SourceDestination(
                        source=path_and_bounding_boxes.path_to_image,
                        destination=destination.joinpath(
                            path_and_bounding_boxes.path_to_image.name
                        ),
                    )
                    for path_and_bounding_boxes in copy_paths
                ],
            )

        return len(copy_paths)

    return None


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
