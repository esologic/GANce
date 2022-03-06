"""
Given some styleGAN networks, load each network and synthesize a number of images with them.
Interesting vectors can be reused with other networks.
"""
import itertools
from functools import partial
from itertools import count
from pathlib import Path
from typing import Callable, Iterator, List, NamedTuple, Optional

import click
import numpy as np
from click_option_group import RequiredAnyOptionGroup, optgroup
from PIL import Image

from gance import faces
from gance.gance_types import RGBInt8ImageType
from gance.hash_file import hash_file
from gance.image_sources.still_image_common import PNG, write_image
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import NETWORK_SUFFIX, MultiNetwork
from gance.synthesis_file import SYNTHESIS_FILE_SUFFIX, read_vector_in_file, write_synthesis_file
from gance.vector_sources.primatives import DEFAULT_RANDOM_SEED, gaussian_data
from gance.vector_sources.vector_types import SingleVector


@click.group()
def cli() -> None:
    """
    Given some styleGAN networks, load each network and synthesize a number of images with them.
    Interesting vectors can be reused with other networks.

    \f

    :return: None
    """


def all_paths(
    dir_of_paths: Optional[str], given_paths: Optional[List[str]], suffix: str
) -> List[Path]:
    """
    Boil a list of paths from a CLI down to an actual list of paths.
    :param dir_of_paths: Files in a know dir.
    :param given_paths: Directly passed paths
    :param suffix: Look for this type of file in the dir.
    :return: A list of the given files.
    """

    output_paths = []

    if dir_of_paths is not None:
        output_paths += list(Path(dir_of_paths).glob(f"*{suffix}"))

    if given_paths is not None:
        output_paths += [Path(network_path) for network_path in given_paths]

    return output_paths


@cli.command()
@optgroup.group(
    "network sources",
    cls=RequiredAnyOptionGroup,
    help="Must provide one or more networks directly, or a directory full of networks.",
)
@optgroup.option(
    "--networks-dir",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    default=None,
    help="Path to the directory containing one or more network .pkl file.",
)
@optgroup.option(
    "--network",
    type=click.Path(file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    default=None,
    multiple=True,
    help="Path to a network .pkl file directly.",
)
@optgroup.group(
    "Synthesis file sources",
    cls=RequiredAnyOptionGroup,
    help=(
        "Must provide one or more synthesis files directly, "
        "or a directory full of synthesis files."
    ),
)
@optgroup.option(
    "--synthesis-files-dir",
    type=click.Path(file_okay=False, dir_okay=True, readable=True, resolve_path=True),
    default=None,
    help="Path to a directory that contains one or more synthesis .json files.",
)
@optgroup.option(
    "--synthesis-file",
    type=click.Path(file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    default=None,
    multiple=True,
    help="Path to a synthesis .json file directly.",
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, writable=True, dir_okay=True, resolve_path=True),
    help=(
        "For each of the networks in `networks_directory` a new directory is created here that"
        "contain the desired images."
    ),
)
def synthesis_file_into_networks(  # pylint: disable=too-many-locals
    networks_dir: Optional[str],
    network: Optional[List[str]],
    synthesis_files_dir: Optional[str],
    synthesis_file: Optional[List[str]],
    output_directory: str,
) -> None:
    """
    For each of the input synthesis files (so any .json file with a "vector" key), read the vector.
    For each of those vectors, write it to each of the networks. The resulting synthesis files and
    images are written to directories in `output_directory` named after the input vector file.

    \f -- For Click

    :param networks_dir: See click help.
    :param network: See click help.
    :param synthesis_files_dir: See click help.
    :param synthesis_file: See click help.
    :param output_directory: See click help.
    :return: None
    """

    network_paths = all_paths(dir_of_paths=networks_dir, given_paths=network, suffix=NETWORK_SUFFIX)

    synthesis_file_paths = all_paths(
        dir_of_paths=synthesis_files_dir, given_paths=synthesis_file, suffix=SYNTHESIS_FILE_SUFFIX
    )

    if not network_paths or not synthesis_file_paths:
        LOGGER.info(
            f"No input! "
            f"Found networks: {network_paths} "
            f"Found Syn files: {synthesis_file_paths}"
        )
        return None
    else:
        LOGGER.info(
            f"Found {len(network_paths)} networks and {len(synthesis_file_paths)} synthesis files."
        )

    top_level_output_directory = Path(output_directory)
    top_level_output_directory.mkdir(exist_ok=True)

    def _create_dir(path: Path) -> Path:
        """
        Helper function, creates a directory at the given path.
        :param path: Path to directory.
        :return: Input path but now the underlying directory has been created.
        """
        path.mkdir(exist_ok=True)
        return path

    output_directories = [
        _create_dir(top_level_output_directory.joinpath(path.with_suffix("").name))
        for path in synthesis_file_paths
    ]

    synthesis_file_hashes = [hash_file(path) for path in synthesis_file_paths]

    vectors = [read_vector_in_file(path) for path in synthesis_file_paths]

    with MultiNetwork(network_paths=network_paths) as multi_network:

        if multi_network is None:
            LOGGER.error("Couldn't load network")
            return None

        # I know this loop looks like it's in the wrong order, but it's not!
        # We only want to switch networks for as many networks as their are because the operation
        # is expensive, that's why we're doing the double loop here.
        for index, network_path in enumerate(network_paths):
            network_hash = hash_file(network_path)
            for synthesis_file_hash, vector, directory in zip(
                synthesis_file_hashes, vectors, output_directories
            ):
                image_path = directory.joinpath(
                    f"network_{network_path.name}_{synthesis_file_hash}.{PNG}"
                )

                image = Image.fromarray(
                    multi_network.indexed_create_image_vector(index=index, data=vector)
                )
                image.save(fp=str(image_path), format=PNG.upper())

                write_synthesis_file(
                    destination_path=image_path.with_suffix(SYNTHESIS_FILE_SUFFIX),
                    network_path=network_path,
                    network_hash=network_hash,
                    image_path=image_path,
                    image_hash=hash_file(image_path),
                    vector=vector,
                )

    return None


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


@cli.command()
@click.option(
    "--networks-directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Path to the directory containing one or more network .pkl file.",
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="For each of the networks in `networks_directory` a new directory is created here that"
    "contain the desired images.",
)
@click.option(
    "--num-faces",
    type=click.IntRange(min=0),
    help="For each network, this number of images that contain faces will be created",
)
@click.option(
    "--no-faces",
    type=click.IntRange(min=0),
    help="For each network, this number of images that do not contain faces will be created",
)
@click.option(
    "--random-seed",
    type=click.INT,
    help="For each network, this number of images that do not contain faces will be created",
    default=DEFAULT_RANDOM_SEED,
    show_default=True,
)
def images_from_network(  # pylint: disable=too-many-locals
    networks_directory: str, output_directory: str, num_faces: int, no_faces: int, random_seed: int
) -> None:
    """
    Given a directory of networks, load each network and synthesize a number of images with them.
    Set how many images you want with or without faces. Results are written per network into a set
    of nested directories. For each image, a `.json` file is also written to describe how the
    image as created so it can be re-created or slightly tweaked.

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
    cli()  # pylint: disable=no-value-for-parameter
