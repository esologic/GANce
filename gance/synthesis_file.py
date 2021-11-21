"""
Functions to write and read "synthesis files", which are json files that describe a vector that
was fed into a model, the model that was used, and the resulting image.
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


@dataclass_json
@dataclass
class SynthesisFileDict:
    """
    Properties of a synthesis run, can be loaded later to figure out how an image was created.
    """

    # The vector that was input to the model to produce the resulting image associated with this
    # file.
    vector: Union[List[List[float]], List[List[List[float]]]]

    # Path to the model used in the synthesis. Will changed! Included to give some context to
    # future reader.
    model_path: str

    # Hash of the model used in synthesis.
    model_hash: str

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
    model_path: Path,
    model_hash: str,
    image_path: Path,
    image_hash: str,
) -> None:
    """
    Wrapper to write a synthesis file to disk.
    :param destination_path: Path to write the file to.
    :param vector: See docs in dataclass.
    :param model_path: See docs in dataclass.
    :param model_hash: See docs in dataclass.
    :param image_path: See docs in dataclass.
    :param image_hash: See docs in dataclass.
    :return: None
    """

    with open(str(destination_path), "w") as file:
        json.dump(
            SynthesisFileDict(  # type: ignore # pylint: disable=no-member
                model_path=str(model_path),
                model_hash=model_hash,
                image_path=str(image_path),
                image_hash=image_hash,
                vector=vector.tolist(),
                version=Version.version_1,
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

        loaded = SynthesisFileDict.from_dict(  # type: ignore # pylint: disable=no-member
            json.load(file)
        )

        version = loaded.version if loaded.version is not None else Version.version_0

        vector = np.array(loaded.vector)

        if version == Version.version_0:
            # Bug in early version, accidentally stored input vectors in the model input form.
            # Now we do that with the interface.
            vector = vector[0]

        return SingleVector(vector)
