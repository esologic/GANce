"""
Given the cropped images, scale them to sizes usable by stylegan
"""

import multiprocessing
from functools import partial
from pathlib import Path

import click
import cv2

DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024


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


@click.command()
@click.option(
    "--original_images_directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help="A directory containing the images to scale.",
)
@click.option(
    "--image_file_extension",
    type=str,
    default="jpeg",
    help="The file extension for the images in original_images_directory.",
    show_default=True,
)
@click.option(
    "--output_directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True),
    help="The directory the scaled images will be placed in.",
)
@click.option(
    "--scaled_image_width",
    type=click.IntRange(min=0),
    default=DEFAULT_IMAGE_WIDTH,
    help="The width of the scaled image.",
    show_default=True,
)
@click.option(
    "--scaled_image_height",
    type=click.IntRange(min=0),
    default=DEFAULT_IMAGE_HEIGHT,
    help="The height of the scaled image.",
    show_default=True,
)
def scales_images(
    original_images_directory: str,
    image_file_extension: str,
    output_directory: str,
    scaled_image_width: int,
    scaled_image_height: int,
) -> None:
    """
    Scale images using multiple processes.
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


if __name__ == "__main__":
    scales_images()  # pylint: disable=no-value-for-parameter
