"""
Test suite of projection related functions.
"""

import shutil
from pathlib import Path
from test.assets import BATCH_2_IMAGE_1_PATH, BATCH_2_IMAGE_2_PATH

import pytest

from gance.image_sources.still_image_common import read_image
from gance.projection.projector_file_writer import _image_to_tfrecords_directory
from gance.stylegan2.dataset_tool import compare, create_from_images


@pytest.mark.parametrize(
    "image_a,image_b,expected_result",
    [
        (BATCH_2_IMAGE_1_PATH, BATCH_2_IMAGE_2_PATH, False),
        (BATCH_2_IMAGE_1_PATH, BATCH_2_IMAGE_1_PATH, True),
    ],
)
def test_image_to_tfrecords_directory(
    tmpdir: str, image_a: Path, image_b: Path, expected_result: bool
) -> None:
    """
    Checks to make sure that the library version of the `.tfrecords` conversion works the same
    as the one used in YAC.
    :return: None
    """

    temp_dir = Path(tmpdir)

    test_directory = temp_dir.joinpath("test")
    test_directory.mkdir()

    _image_to_tfrecords_directory(
        records_directory=test_directory,
        image=read_image(image_a),
    )

    good_directory = temp_dir.joinpath("good")
    good_directory.mkdir()

    images_directory = temp_dir.joinpath("images")
    images_directory.mkdir()
    shutil.copyfile(src=image_b, dst=images_directory.joinpath(image_b.name))

    create_from_images(  # type: ignore
        tfrecord_dir=str(good_directory),
        image_dir=str(images_directory),
        shuffle=False,
    )

    equal = compare(  # type: ignore
        tfrecord_dir_a=test_directory, tfrecord_dir_b=good_directory, ignore_labels=True
    )

    assert equal == expected_result
