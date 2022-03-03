"""
CLI to project videos to a file
"""

from pathlib import Path
from typing import List, Optional

import click
from click_option_group import RequiredAnyOptionGroup, optgroup
from PIL import Image

from gance.hash_file import hash_file
from gance.image_sources.still_image_common import PNG
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import NETWORK_SUFFIX, MultiNetwork
from gance.synthesis_file import SYNTHESIS_FILE_SUFFIX, read_vector_in_file, write_synthesis_file


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


@click.command()
@optgroup.group(
    "network sources",
    cls=RequiredAnyOptionGroup,
    help="Must provide one or more networks directly, or a directory full of networks.",
)
@optgroup.option(
    "--networks_dir",
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
    "--synthesis_files_dir",
    type=click.Path(file_okay=False, dir_okay=True, readable=True, resolve_path=True),
    default=None,
    help="Path to a directory that contains one or more synthesis .json files.",
)
@optgroup.option(
    "--synthesis_file",
    type=click.Path(file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    default=None,
    multiple=True,
    help="Path to a synthesis .json file directly.",
)
@click.option(
    "--output_directory",
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


if __name__ == "__main__":
    synthesis_file_into_networks()  # pylint: disable=no-value-for-parameter
