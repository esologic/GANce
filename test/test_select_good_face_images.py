"""
Unit tests around selection of face images to be input to training.
"""

import datetime
import itertools
from pathlib import Path
from test.assets import FACE_IMAGE_PATH, NO_FACE_IMAGE_PATH
from typing import List, Optional, Tuple

import pytest
from py._path.local import LocalPath

from gance import pi_images_common, select_good_face_images


def create_images_in_directory(
    parent_directory: Path, dataset_name: str, image_path: Path, num_images: int
) -> Tuple[Path, List[Path]]:
    """
    Wrapper to copy a given image n times into a directory.
    :param parent_directory: The path to the parent directory to create this new directory
    of images in.
    :param dataset_name: Used for the directory name and the file names.
    :param image_path: The path to the source image.
    :param num_images: The num copies of the image to make.
    :return: A Tuple, (directory with images, list of paths to images)
    """

    directory = parent_directory.joinpath(dataset_name)
    directory.mkdir()

    return (
        directory,
        make_copies_of_image(
            destination_directory=directory,
            source_file=image_path,
            num_copies=num_images,
            dataset_name=dataset_name,
        ),
    )


@pytest.mark.parametrize(
    (
        "num_face_images_primary,"
        "num_no_face_images_primary,"
        "num_face_images_secondary,"
        "num_no_face_images_secondary,"
        "target_num_images"
    ),
    [
        (1, 1, 1, 0, 3),
        (1, 1, 0, 1, 3),
        (1, 0, 1, 1, 3),
        (0, 1, 1, 1, 3),
        (3, 0, 0, 0, 3),
        (0, 3, 0, 0, 3),
        (0, 0, 3, 0, 3),
        (0, 0, 0, 3, 3),
    ],
)
def test_select_images_for_training(  # pylint: disable=too-many-locals
    tmpdir: LocalPath,
    num_face_images_primary: int,
    num_no_face_images_primary: int,
    num_face_images_secondary: int,
    num_no_face_images_secondary: int,
    target_num_images: int,
) -> None:
    """
    Makes sure the selection procedure works as expected using real images.
    Creates four directories, two passed as primary and two passed as secondary.
    :param tmpdir: Test fixture.
    :param num_face_images_primary: Num images containing a face to be put in the primary, with
    faces directory.
    :param num_no_face_images_primary: Num images that do not contain a face to be put in the
    primary, without faces directory.
    :param num_face_images_secondary: Num images containing a face to be put in the secondary, with
    faces directory.
    :param num_no_face_images_secondary: Num images that do not contain a face to be put in the
    secondary, without faces directory.
    :param target_num_images: The desired num images to select from the primary/secondary
    directories.
    :return: None
    """

    tmpdir_path = Path(tmpdir)

    face_images_primary, no_face_images_primary, face_images_secondary, no_face_images_secondary = [
        create_images_in_directory(tmpdir_path, dataset_name, image_path, num_images)
        for dataset_name, image_path, num_images in [
            ("face_images_primary", FACE_IMAGE_PATH, num_face_images_primary),
            ("no_face_images_primary", NO_FACE_IMAGE_PATH, num_no_face_images_primary),
            ("face_images_secondary", FACE_IMAGE_PATH, num_face_images_secondary),
            ("no_face_images_secondary", NO_FACE_IMAGE_PATH, num_no_face_images_secondary),
        ]
    ]

    output = select_good_face_images.select_images_for_training(
        primary_directory=[str(face_images_primary[0]), str(no_face_images_primary[0])],
        secondary_directory=[str(face_images_secondary[0]), str(no_face_images_secondary[0])],
        target_num_images=target_num_images,
    )

    # Make sure we only iterate through the output list once.
    output_in_order = iter(output.path_and_bounding_boxes)

    # Order matters for the first part of the list
    assert all(
        expected_image == output_image.path_to_image
        for output_image, expected_image in zip(
            output_in_order, itertools.chain(face_images_primary[1], face_images_secondary[1])
        )
    )

    # Order doesn't matter with this half of the list
    randomly_selected_images = set(
        itertools.chain(no_face_images_primary[1], no_face_images_secondary[1])
    )
    assert all(
        (output_image.path_to_image in randomly_selected_images) for output_image in output_in_order
    )

    num_available_images = (
        num_face_images_primary
        + num_face_images_secondary
        + num_no_face_images_primary
        + num_no_face_images_secondary
    )
    num_output_images = len(output.path_and_bounding_boxes)

    assert (
        (num_output_images == target_num_images)
        if (num_available_images > target_num_images)
        else num_output_images == num_available_images
    )


@pytest.mark.parametrize(
    "file_name,timestamp",
    [
        (
            "april_27_cottage_session_1_04-28-2021_11-48-52-507461",
            datetime.datetime(
                year=2021, month=4, day=28, hour=11, minute=48, second=52, microsecond=507461
            ),
        ),
        (
            "april_27_cottage_session_1_04-28-2021_11-50-12-752379",
            datetime.datetime(
                year=2021, month=4, day=28, hour=11, minute=50, second=12, microsecond=752379
            ),
        ),
        (
            "april_27_cottage_session_1_04-28-2021_11-50-48-250746",
            datetime.datetime(
                year=2021, month=4, day=28, hour=11, minute=50, second=48, microsecond=250746
            ),
        ),
    ],
)
def test_parse_timestamp_from_filename(file_name: str, timestamp: datetime.datetime) -> None:
    """
    Tests to make sure the parsing function works using some real names.
    :param file_name: The string to parse.
    :param timestamp: The expected parse result.
    :return: None
    """
    assert select_good_face_images.parse_timestamp_from_filename(file_name) == timestamp


def make_copies_of_image(
    destination_directory: Path,
    num_copies: int,
    source_file: Path,
    dataset_name: Optional[str] = None,
) -> List[Path]:
    """
    Helper function, makes `num_copies` copies of `source_file` in `destination_directory`.
    :param destination_directory: The location to copy the file to.
    :param num_copies: The num copies to make.
    :param source_file: The image to copy.
    :param dataset_name: If given, images will be created with this dataset name. If not given,
    the dataset name is the name of the image file without an extension.
    :return: The list of the new files.
    """

    destinations = [
        destination_directory.joinpath(
            pi_images_common.create_image_filename(
                dataset_name=dataset_name
                if dataset_name is not None
                else source_file.with_suffix("").name,
                capture_time=datetime.datetime.now(),
            )
        )
        for _ in range(num_copies)
    ]

    for path in destinations:
        select_good_face_images.copy(
            select_good_face_images.SourceDestination(source=source_file, destination=path)
        )

    return destinations


@pytest.mark.parametrize(
    "num_face_images,num_no_face_images", [(1, 0), (0, 1), (1, 1), (10, 0), (0, 10), (10, 10)]
)
def test__scan_images_in_directories(
    tmpdir: LocalPath, num_face_images: int, num_no_face_images: int
) -> None:
    """
    Test the image scanning function using real images.
    :param tmpdir: Test fixture.
    :param num_face_images: The num images with faces to create/detect.
    :param num_no_face_images:The num images without faces to create/detect.
    :return: None
    """
    tmpdir_path = Path(tmpdir)

    faces_directory, _ = create_images_in_directory(
        tmpdir_path, FACE_IMAGE_PATH.with_suffix("").name, FACE_IMAGE_PATH, num_face_images
    )
    no_faces_directory, _ = create_images_in_directory(
        tmpdir_path, NO_FACE_IMAGE_PATH.with_suffix("").name, NO_FACE_IMAGE_PATH, num_no_face_images
    )

    (
        result_faces,
        result_no_faces,
    ) = select_good_face_images._scan_images_in_directories(  # pylint: disable=protected-access
        [faces_directory, no_faces_directory]
    )

    assert all(
        select_good_face_images._contains_face(result) is True  # pylint: disable=protected-access
        for result in result_faces
    )
    assert all(
        select_good_face_images._contains_face(result) is False  # pylint: disable=protected-access
        for result in result_no_faces
    )
