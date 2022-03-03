"""
Functions to write and read "synthesis files", which are json files that describe a vector that
was fed into a network, the network that was used, and the resulting image.
"""

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from dataclasses_json import dataclass_json

from gance.vector_sources.vector_types import SingleVector

SYNTHESIS_FILE_SUFFIX = ".json"


class Version(IntEnum):
    """
    Contains synthesis file versions.
    """

    version_0 = 0
    version_1 = 1

    # Renamed fields with 'model' in them to use 'network' instead.
    version_2 = 2


@dataclass_json
@dataclass
class SynthesisFileDict:
    """
    Properties of a synthesis run, can be loaded later to figure out how an image was created.
    """

    # The vector that was input to the network to produce the resulting image associated with this
    # file.
    vector: Union[List[List[float]], List[List[List[float]]]]

    # Path to the network used in the synthesis. Will changed! Included to give some context to
    # future reader.
    network_path: str

    # Hash of the network used in synthesis.
    network_hash: str

    # Original path to this file's pair. This will also move but again is included to give some
    # context.
    image_path: str

    # Hash of this file's pair.
    image_hash: str

    # Version of the file, see type docs.
    version: Optional[Version] = None


def write_synthesis_file(
    destination_path: Path,
    vector: SingleVector,
    network_path: Path,
    network_hash: str,
    image_path: Path,
    image_hash: str,
) -> None:
    """
    Wrapper to write a synthesis file to disk.
    :param destination_path: Path to write the file to.
    :param vector: See docs in dataclass.
    :param network_path: See docs in dataclass.
    :param network_hash: See docs in dataclass.
    :param image_path: See docs in dataclass.
    :param image_hash: See docs in dataclass.
    :return: None
    """

    with open(str(destination_path), "w") as file:
        json.dump(
            SynthesisFileDict(  # type: ignore # pylint: disable=no-member
                network_path=str(network_path),
                network_hash=network_hash,
                image_path=str(image_path),
                image_hash=image_hash,
                vector=vector.tolist(),
                version=Version.version_2,
            ).to_dict(),
            file,
        )


def read_vector_in_file(path_to_json: Path) -> SingleVector:
    """
    Read the vector (only the vector) in a given synthesis file.
    :param path_to_json: Path to the file.
    :return: Vector
    """

    with open(str(path_to_json), "r") as file:

        raw_dict = json.load(file)

        # No 'version' key means it's version 0.
        if 'version' not in raw_dict or raw_dict['version'] < Version.version_2:
            raw_dict['network_path'] = raw_dict.pop("model_path")
            raw_dict['network_hash'] = raw_dict.pop("model_hash")

        loaded = SynthesisFileDict.from_dict(  # type: ignore # pylint: disable=no-member
            raw_dict
        )

        version = loaded.version if loaded.version is not None else Version.version_0

        vector = np.array(loaded.vector)

        if version == Version.version_0:
            # Bug in early version, accidentally stored input vectors in the network input form.
            # Now we do that with the interface.
            vector = vector[0]

        return SingleVector(vector)
