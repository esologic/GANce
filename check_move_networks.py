"""
CLI for validating and copying networks.
"""

import shutil
from itertools import count
from pathlib import Path
from typing import List

import click

from gance.hash_file import hash_file
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import NETWORK_SUFFIX, MultiNetwork
from gance.vector_sources.primatives import gaussian_data


@click.command()
@click.option(
    "--network-directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Directory of networks. Will be passed in order.",
    multiple=True,
    required=True,
)
@click.option(
    "--output-directory",
    type=click.Path(file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Destination directory for validated networks.",
    required=True,
)
def check_move_networks(network_directory: List[str], output_directory: str) -> None:
    """
    Scan directories of networks for pickled networks, for each of these networks, load them, and
    feed in a vector to make sure they are still functional (no bit rot).
    Valid networks are copied to the output directory and renamed with the
    pattern: `{count}_{parent_dir_name}_{network_name}`.

    \f

    :param network_directory: See click docs.
    :param output_directory: See click docs.
    :return: None
    """

    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(exist_ok=True)

    network_count = count()

    for network_directory_path in (Path(directory) for directory in sorted(network_directory)):
        LOGGER.info(f"Scanning for networks in: {network_directory_path}")

        network_paths = list(sorted(network_directory_path.glob(f"*{NETWORK_SUFFIX}")))

        with MultiNetwork(network_paths=network_paths) as multi_network:
            vector = gaussian_data(
                num_vectors=1, vector_length=multi_network.expected_vector_length
            ).reshape((1, multi_network.expected_vector_length))
            for network_index, original_network_path in enumerate(network_paths):

                network_destination_path = output_directory_path.joinpath(
                    f"{next(network_count)}_"
                    f"{original_network_path.parent.name}_{original_network_path.name}"
                )

                if network_destination_path.exists() and hash_file(
                    network_destination_path
                ) == hash_file(original_network_path):
                    LOGGER.info(
                        f"network had already been copied, "
                        f"skipping verification of: {original_network_path}"
                    )
                elif original_network_path.name == "submit_config.pkl":
                    LOGGER.info(f"Skipping pkl, not a network: {original_network_path}")
                else:
                    try:
                        multi_network.indexed_create_image_vector(index=network_index, data=vector)
                        shutil.copyfile(
                            src=str(original_network_path), dst=str(network_destination_path)
                        )
                        LOGGER.info(
                            f"Copied network {original_network_path} -> {network_destination_path}"
                        )
                    except Exception:  # pylint: disable=broad-except
                        LOGGER.exception(
                            f"Something went wrong validating network, not copying to output. "
                            f"network path: {original_network_path}"
                        )


if __name__ == "__main__":
    check_move_networks()  # pylint: disable=no-value-for-parameter
