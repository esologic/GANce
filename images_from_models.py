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
from skimage import data
from skimage.feature import Cascade

from gance.gance_types import RGBInt8ImageType
from gance.hash_file import hash_file
from gance.image_sources.still_image_common import PNG, write_image
from gance.logger_common import LOGGER
from gance.model_interface.model_functions import MODEL_SUFFIX, MultiModel
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

    # One of the greatest evils in this whole repo.
    # Under the hood, `face_recognition` uses dnnlib, which stylegan also uses.
    # An init function is called upon import, which makes the loading of models within the
    # subprocesses impossible. There's probably a better way around this but this was expedient.
    # import face_recognition

    num_images = count()

    face_detector = Cascade(data.lbp_frontal_face_cascade_filename())

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
                face_detector.detect_multi_scale(
                    img=image,
                    scale_factor=1.9,
                    step_ratio=1,
                    min_size=(1, 1),
                    max_size=tuple(image.shape[:2]),
                )
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
    model_name: str,
    model_path: Path,
    model_hash: str,
) -> None:
    """
    Pull images from the iterator, write them to disk.
    For each image, a `.json` file is also written with info about how it was generated.
    :param images: Image source.
    :param output_directory: Destination directory of images.
    :param model_name: For the filename and context file.
    :param model_path: For the filename and context file.
    :param model_hash: For the filename and context file.
    :return: None
    """

    for image_index, output in enumerate(images):

        image_path = output_directory.joinpath(
            f"{model_name}_{model_hash}_"
            f"{'face' if output.contains_face else 'no_face'}_{image_index}.{PNG}"
        )

        write_image(image=output.image, path=image_path)

        write_synthesis_file(
            destination_path=image_path.with_suffix(".json"),
            model_path=model_path,
            model_hash=model_hash,
            image_path=image_path,
            image_hash=hash_file(image_path),
            vector=output.vector,
        )

        LOGGER.info(f"Wrote image {image_path}")


@click.command()
@click.option(
    "--models_directory",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Path to the directory containing one or more model .pkl file.",
)
@click.option(
    "--output_directory",
    type=click.Path(file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="For each of the models in `models_directory` a new directory is created here that"
    "contain the desired images.",
)
@click.option(
    "--faces",
    type=click.IntRange(min=0),
    help="For each model, this number of images that contain faces will be created",
)
@click.option(
    "--no_faces",
    type=click.IntRange(min=0),
    help="For each model, this number of images that do not contain faces will be created",
)
@click.option(
    "--random_seed",
    type=click.INT,
    help="For each model, this number of images that do not contain faces will be created",
    default=DEFAULT_RANDOM_SEED,
    show_default=True,
)
def images_from_model(  # pylint: disable=too-many-locals
    models_directory: str, output_directory: str, faces: int, no_faces: int, random_seed: int
) -> None:
    """
    Given a directory of models, create random images from them. Set how many images you want
    with or without faces. Results are written per model into a set of nested directories.
    For each image, a `.json` file is also written to describe how the image as created so it
    could be re-created.

    \f -- For Click

    :param models_directory: See click help.
    :param output_directory: See click help.
    :param faces: See click help.
    :param no_faces: See click help.
    :param random_seed: See click help.
    :return: None
    """

    model_paths = list(Path(models_directory).glob(f"*{MODEL_SUFFIX}"))
    top_level_output_directory = Path(output_directory)
    top_level_output_directory.mkdir(exist_ok=True)

    if not model_paths:
        LOGGER.info(f"Couldn't find any {MODEL_SUFFIX} files in {models_directory}. Exiting.")
        return None
    else:
        LOGGER.info(f"Found {len(model_paths)} models.")

    with MultiModel(model_paths=model_paths) as multi_model:

        if multi_model is None:
            LOGGER.error("Couldn't load model")
            return None

        random_state = np.random.RandomState(random_seed)  # pylint: disable=no-member

        for index, model_path in enumerate(model_paths):

            LOGGER.info(f"Writing images for {model_path}")

            model_hash = hash_file(model_path)

            model_name = model_path.with_suffix("").name

            current_output_directory = top_level_output_directory.joinpath(model_name)
            current_output_directory.mkdir(exist_ok=True)

            for contains_face, num_images in [(True, faces), (False, no_faces)]:
                write_images(
                    images=itertools.islice(
                        create_images(
                            image_function=partial(multi_model.indexed_create_image_vector, index),
                            contains_face=contains_face,
                            random_state=random_state,
                            vector_length=multi_model.expected_vector_length,
                        ),
                        num_images,
                    ),
                    model_hash=model_hash,
                    model_name=model_name,
                    model_path=model_path,
                    output_directory=current_output_directory,
                )

    return None


if __name__ == "__main__":
    images_from_model()  # pylint: disable=no-value-for-parameter
