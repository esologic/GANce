"""
CLI for validating and copying models.
"""

import shutil
from itertools import count
from pathlib import Path
from typing import List

import click

from gance.hash_file import hash_file
from gance.logger_common import LOGGER
from gance.model_interface.model_functions import MODEL_SUFFIX, MultiModel
from gance.vector_sources.primatives import gaussian_data


@click.command()
@click.option(
    "--model_directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Directory of models. Will be passed in order.",
    multiple=True,
    required=True,
)
@click.option(
    "--output_directory",
    type=click.Path(file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Destination directory for validated models.",
    required=True,
)
def check_move_models(model_directory: List[str], output_directory: str) -> None:
    """
    Scan directories of models for model files, for each of these models, load them, and feed in a
    vector to make sure they are valid and work for our purposes. Valid models are copied to the
    output directory and renamed with the pattern: `{count}_{parent_dir_name}_{model_name}`.

    \f

    :param model_directory: See click docs.
    :param output_directory: See click docs.
    :return: None
    """

    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(exist_ok=True)

    model_count = count()

    for model_directory_path in (Path(directory) for directory in sorted(model_directory)):
        LOGGER.info(f"Scanning for models in: {model_directory_path}")

        model_paths = list(sorted(model_directory_path.glob(f"*{MODEL_SUFFIX}")))

        with MultiModel(model_paths=model_paths) as multi_model:
            vector = gaussian_data(
                num_vectors=1, vector_length=multi_model.expected_vector_length
            ).reshape((1, multi_model.expected_vector_length))
            for model_index, original_model_path in enumerate(model_paths):

                model_destination_path = output_directory_path.joinpath(
                    f"{next(model_count)}_"
                    f"{original_model_path.parent.name}_{original_model_path.name}"
                )

                if model_destination_path.exists() and hash_file(
                    model_destination_path
                ) == hash_file(original_model_path):
                    LOGGER.info(
                        f"Model had already been copied, "
                        f"skipping verification of: {original_model_path}"
                    )
                elif original_model_path.name == "submit_config.pkl":
                    LOGGER.info(f"Skipping pkl, not a model: {original_model_path}")
                else:
                    try:
                        multi_model.indexed_create_image_vector(index=model_index, data=vector)
                        shutil.copyfile(
                            src=str(original_model_path), dst=str(model_destination_path)
                        )
                        LOGGER.info(
                            f"Copied model {original_model_path} -> {model_destination_path}"
                        )
                    except Exception:  # pylint: disable=broad-except
                        LOGGER.exception(
                            f"Something went wrong validating model, not copying to output. "
                            f"Model path: {original_model_path}"
                        )


if __name__ == "__main__":
    check_move_models()  # pylint: disable=no-value-for-parameter
