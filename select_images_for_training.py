"""
Script to select images for training giving a large pool of candidate images.
Read the docs for `select_images_copy` to understand how images are selected.
"""

import json
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

import click

from gance.logger_common import LOGGER
from gance.select_good_face_images import SourceDestination, copy, select_images_for_training


@click.command()
@click.option(
    "--primary_directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help=(
        "Directory of images to search. Images that contain faces are highest priority and will be "
        "selected over all other images in other directories."
    ),
    multiple=True,
    required=True,
)
@click.option(
    "--secondary_directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help=(
        "Directory of images to search. Images that contain faces will be selected if there aren't "
        "enough faces in primary directories to reach the value given by --target_num_images."
    ),
    multiple=True,
)
@click.option(
    "--target_num_images",
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
    "--destination_directory",
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
    select_images_copy()  # pylint: disable=no-value-for-parameter
