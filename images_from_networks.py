"""
CLI to project videos to a file
"""

import itertools
from functools import partial
from itertools import count
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

import click
import numpy as np

from gance import faces
from gance.gance_types import RGBInt8ImageType
from gance.hash_file import hash_file
from gance.image_sources.still_image_common import PNG, write_image
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import NETWORK_SUFFIX, MultiNetwork
from gance.synthesis_file import write_synthesis_file
from gance.vector_sources.primatives import DEFAULT_RANDOM_SEED, gaussian_data
from gance.vector_sources.vector_types import SingleVector


class _ContainsFaceVectorImage(NamedTuple):
    """
    Intermediate type.
    """

    contains_face: bool
    vector: SingleVector
    image: RGBInt8ImageType


def create_images(
    image_function: Callable[[SingleVector], RGBInt8ImageType],
    vector_length: int,
    random_state: "np.random.RandomState",
    contains_face: bool,
) -> Iterator[_ContainsFaceVectorImage]:
    """
    Generator that produces randomized images of the given type.
    :param image_function: Function to call to get an image.
    :param vector_length: Used for randomness source.
    :param random_state: Randomness source.
    :param contains_face: If the resulting image should contain a face or not.
    :return: Info about the image and how it was generated.
    """

    num_images = count()

    face_finder = faces.FaceFinderProxy()

    while True:
        # Random noise vector.
        # Since we're only making 1 vector we can treat it as a `SingleVector`.
        vector: SingleVector = gaussian_data(
            vector_length=vector_length,
            num_vectors=1,
            random_state=random_state,
        )

        image = image_function(vector)
        image_count = next(num_images)

        if (
            any(
                # Constants determined experimentally.
                face_finder.face_locations(image)
            )
            == contains_face
        ):
            LOGGER.info(f"Image #{image_count} met criteria!")
            yield _ContainsFaceVectorImage(contains_face=contains_face, vector=vector, image=image)
        else:
            LOGGER.info(f"Image #{image_count} did not meet criteria.")


def write_images(
    images: Iterator[_ContainsFaceVectorImage],
    output_directory: Path,
    network_name: str,
    network_path: Path,
    network_hash: str,
) -> None:
    """
    Pull images from the iterator, write them to disk.
    For each image, a `.json` file is also written with info about how it was generated.
    :param images: Image source.
    :param output_directory: Destination directory of images.
    :param network_name: For the filename and context file.
    :param network_path: For the filename and context file.
    :param network_hash: For the filename and context file.
    :return: None
    """

    for image_index, output in enumerate(images):

        image_path = output_directory.joinpath(
            f"{network_name}_{network_hash}_"
            f"{'face' if output.contains_face else 'no_face'}_{image_index}.{PNG}"
        )

        write_image(image=output.image, path=image_path)

        write_synthesis_file(
            destination_path=image_path.with_suffix(".json"),
            network_path=network_path,
            network_hash=network_hash,
            image_path=image_path,
            image_hash=hash_file(image_path),
            vector=output.vector,
        )

        LOGGER.info(f"Wrote image {image_path}")


@click.command()
@click.option(
    "--networks_directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Path to the directory containing one or more network .pkl file.",
)
@click.option(
    "--output_directory",
    type=click.Path(file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="For each of the networks in `networks_directory` a new directory is created here that"
    "contain the desired images.",
)
@click.option(
    "--num_faces",
    type=click.IntRange(min=0),
    help="For each network, this number of images that contain faces will be created",
)
@click.option(
    "--no_faces",
    type=click.IntRange(min=0),
    help="For each network, this number of images that do not contain faces will be created",
)
@click.option(
    "--random_seed",
    type=click.INT,
    help="For each network, this number of images that do not contain faces will be created",
    default=DEFAULT_RANDOM_SEED,
    show_default=True,
)
def images_from_network(  # pylint: disable=too-many-locals
    networks_directory: str, output_directory: str, num_faces: int, no_faces: int, random_seed: int
) -> None:
    """
    Given a directory of networks, create random images from them. Set how many images you want
    with or without faces. Results are written per network into a set of nested directories.
    For each image, a `.json` file is also written to describe how the image as created so it
    could be re-created.

    \f -- For Click

    :param networks_directory: See click help.
    :param output_directory: See click help.
    :param num_faces: See click help.
    :param no_faces: See click help.
    :param random_seed: See click help.
    :return: None
    """

    network_paths = list(Path(networks_directory).glob(f"*{NETWORK_SUFFIX}"))
    top_level_output_directory = Path(output_directory)
    top_level_output_directory.mkdir(exist_ok=True)

    if not network_paths:
        LOGGER.info(f"Couldn't find any {NETWORK_SUFFIX} files in {networks_directory}. Exiting.")
        return None
    else:
        LOGGER.info(f"Found {len(network_paths)} networks.")

    with MultiNetwork(network_paths=network_paths) as multi_network:

        if multi_network is None:
            LOGGER.error("Couldn't load network")
            return None

        random_state = np.random.RandomState(random_seed)  # pylint: disable=no-member

        for index, network_path in enumerate(network_paths):

            LOGGER.info(f"Writing images for {network_path}")

            network_hash = hash_file(network_path)

            network_name = network_path.with_suffix("").name

            current_output_directory = top_level_output_directory.joinpath(network_name)
            current_output_directory.mkdir(exist_ok=True)

            for contains_face, num_images in [(True, num_faces), (False, no_faces)]:
                write_images(
                    images=itertools.islice(
                        create_images(
                            image_function=partial(
                                multi_network.indexed_create_image_vector, index
                            ),
                            contains_face=contains_face,
                            random_state=random_state,
                            vector_length=multi_network.expected_vector_length,
                        ),
                        num_images,
                    ),
                    network_hash=network_hash,
                    network_name=network_name,
                    network_path=network_path,
                    output_directory=current_output_directory,
                )

    return None


if __name__ == "__main__":
    images_from_network()  # pylint: disable=no-value-for-parameter
