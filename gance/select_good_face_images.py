"""
Functions used to select images with faces in them from the pool of candidate images.
"""
import datetime
import itertools
import random
import shutil
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator, List, NamedTuple, Optional, Set, Union

import more_itertools

from gance.common_paths import DEVON_FACE_IMAGES_PATH
from gance.faces import FaceFinderProxy
from gance.gance_types import PathAndBoundingBoxes
from gance.image_sources import still_image_common
from gance.logger_common import LOGGER
from gance.pi_images_common import IMAGE_TIMESTAMP_FORMAT


def face_bounding_boxes(
    face_finder: FaceFinderProxy, path_to_image: Path
) -> Optional[PathAndBoundingBoxes]:
    """
    Return a list of bounding boxes for each face in a given image. If no faces are detected,
    return None.
    :param face_finder: Interface for being able to find faces.
    :param path_to_image: The path to the image.
    :return: The list of bounding boxes for each face in the image at the path. Point order for the
    bounding box is (top, right, bottom, left)
    """
    try:
        image = still_image_common.read_image(image_path=path_to_image, mode="RGB")
    except (SyntaxError, OSError):
        LOGGER.error(f"Couldn't open image {path_to_image}")
        return None

    return PathAndBoundingBoxes(
        path_to_image=path_to_image, bounding_boxes=tuple(face_finder.face_locations(image))
    )


def select_good_face_images(candidate_paths: List[Path]) -> List[Path]:
    """
    Given a list of paths to images that could have faces in them, return only those that contain
    faces.
    :param candidate_paths: A list of the candidate images that could have faces in them.
    :return: A list of Named Tuples
    """

    face_finder = FaceFinderProxy()

    # Note: Do not try to make this faster by spreading it across multiple processes.
    # The underlying dlib code is already parallelized using the gpu, and the overhead
    # of breaking up the work decreases throughput.
    paths_and_bounding_boxes = map(partial(face_bounding_boxes, face_finder), candidate_paths)

    # Filter out images that do not contain faces
    contains_faces = filter(
        lambda path_and_bounding_boxes: len(path_and_bounding_boxes.bounding_boxes) > 0,
        paths_and_bounding_boxes,
    )

    # Only return the paths
    return [path_and_bounding_boxes.path_to_image for path_and_bounding_boxes in contains_faces]


def copy_image_to_good(path_to_image_to_copy: Path) -> None:
    """
    Given an image with a face in it, copy it over to the folder designated for these images.
    :param path_to_image_to_copy: The path to the face image.
    :return: None
    """
    shutil.copyfile(
        src=str(path_to_image_to_copy),
        dst=str(DEVON_FACE_IMAGES_PATH.joinpath(path_to_image_to_copy.name)),
    )


class SourceDestination(NamedTuple):
    """
    Mapping a file to be copied to it's new destination path.
    """

    source: Path
    destination: Path


def copy(source_destination: SourceDestination) -> None:
    """
    Copy a file to a destination directory with the same name.
    :param source_destination: Links the file to the directory it's to be copied to.
    :return: None
    """
    source = str(source_destination.source)
    destination = str(source_destination.destination)
    LOGGER.info(f"Copying {source} -> {destination}")
    shutil.copyfile(src=source, dst=destination)


def _images_in_directory(directory: Path) -> List[Path]:
    """
    Lists the image files in a directory.
    This needs to be a standalone function so it can be used with a `multiprocessing.Pool` map.
    :param directory: Path to the directory that contains images.
    :return: A list of the image files in the directory.
    """
    return list(directory.glob("*.jpeg"))


class _PathAndBoundingBoxesAndTimestamp(NamedTuple):
    """
    Intermediate type, adds the capture timestamp from the path.
    """

    path_and_bounding_boxes: PathAndBoundingBoxes
    capture_datetime: datetime.datetime


def parse_timestamp_from_filename(file_name: str) -> datetime.datetime:
    """
    Given an images filename as a string, parse the capture time and return it as a datetime.
    :param file_name: The images filename as a string
    ex: `april_27_cottage_session_1_04-28-2021_11-48-52-507461`. Make sure the suffix, `.png`,
    `.jpeg` is dropped.
    :return: The capture timestamp as a datetime.
    """
    underscore_locations = [index for index, character in enumerate(file_name) if character == "_"]
    # The `+1` is to ensure that the underscore itself is dropped
    datetime_string = file_name[underscore_locations[-2] + 1 :]
    return datetime.datetime.strptime(datetime_string, IMAGE_TIMESTAMP_FORMAT)


def _sort_images_by_filename(images: List[PathAndBoundingBoxes]) -> List[PathAndBoundingBoxes]:
    """
    For each image in an input list, parse out the creation time from the file name, then return
    a list of these files in the order they were created.
    :param images: Images to sorted.
    :return: Sorted images.
    """
    with_timestamp: Iterator[_PathAndBoundingBoxesAndTimestamp] = map(
        lambda p: _PathAndBoundingBoxesAndTimestamp(
            path_and_bounding_boxes=p,
            capture_datetime=parse_timestamp_from_filename(p.path_to_image.with_suffix("").name),
        ),
        images,
    )
    sorted_by_timestamp = sorted(with_timestamp, key=lambda p: p.capture_datetime)

    # Drop the timestamp fields before returning.
    return list(map(lambda p: p.path_and_bounding_boxes, sorted_by_timestamp))


def _scan_images_in_directories(directories: Iterable[Path]) -> List[List[PathAndBoundingBoxes]]:
    """
    Find face bounding boxes in the images in a given list (iterator) of directories.
    Note to future Devon: You may be tempted to use a multiprocessing pool here to try and speed
    things up here. Under the hood, dlib makes decisions to give you the best results for your
    platform (cpu, GPU) etc, so the `face_recognition` library is going to give you the most
    effecient search. I did some experimentation, and found that using a regular `map` here
    vs a `Pool().map` was 33% faster.
    :param directories: The directories to scan.
    :return: A 1D iterator of the scanned images.
    """

    face_finder = FaceFinderProxy()

    return [
        list(
            filter(
                lambda image: image is not None,
                map(partial(face_bounding_boxes, face_finder), _images_in_directory(directory)),
            )
        )
        for directory in directories
    ]


def _images_around_faces(
    all_possible: Iterable[PathAndBoundingBoxes],
    images_with_faces: Set[PathAndBoundingBoxes],
) -> Iterator[Union[PathAndBoundingBoxes, None]]:
    """
    Selects images adjacent to images that are known to have faces them from the master list of
    images.
    :param all_possible: The master list of images. All images in `images_with_faces` will
    also be in here as well as the images that do not have faces in them.
    :param images_with_faces: A set of images that are known to have faces in them.
    :return: An iterator of the images two before and two after images with faces in them in
    `all_possible`.
    """

    for window in more_itertools.windowed(seq=all_possible, n=5):
        before, value, after = window[:2], window[2], window[3:]
        if value in images_with_faces:
            yield from filter(
                lambda path_and_bounding_boxes: path_and_bounding_boxes not in images_with_faces,
                before + after,
            )


def _contains_face(path_and_bounding_boxes: PathAndBoundingBoxes) -> bool:
    """
    Helper function to determine if an image has a face in it based on the bounding boxes.
    :param path_and_bounding_boxes: The image to run this computation on.
    :return: If there's an image or not.
    """
    output = len(path_and_bounding_boxes.bounding_boxes) > 0
    LOGGER.debug(f"Face in {path_and_bounding_boxes.path_to_image} ? {output}")
    return output


class GenerateWithCount:
    """
    Return items from an input iterator, but keep track of how many items were returned over time
    """

    def __init__(self: "GenerateWithCount", iterator: Iterator[PathAndBoundingBoxes]) -> None:
        """
        :param iterator: Items from this iterator will be returned and counted.
        """
        self._count = 0
        self._iterator = iterator

    def __iter__(self: "GenerateWithCount") -> Iterator[PathAndBoundingBoxes]:
        """
        Yield each of the items in the input iterator, incrementing the counter each time a new
        item is returned.
        :return: An individual item from the input iterator.
        """
        for item in self._iterator:
            self._count += 1
            yield item

    @property
    def count(self: "GenerateWithCount") -> int:
        """
        Getter method, note you can't set this value unprotected.
        :return: The amount of times the iterator has been `next()`-ed.
        """
        return self._count


class ImageSelectionOutput(NamedTuple):
    """
    Result and counts of the image selection process.
    """

    path_and_bounding_boxes: List[PathAndBoundingBoxes]
    num_img_w_faces: int
    num_img_around_img_w_faces: int
    num_randomized_img_wout_faces: int


def select_images_for_training(
    primary_directory: List[str], secondary_directory: List[str], target_num_images: Optional[int]
) -> ImageSelectionOutput:
    """
    (Docs copied from caller)

    User provides the target number of images to select, and
    primary/secondary directories full of images to select from.

    To reach the target number of images, images are selected in this order:

    1. Images that contain faces in primary directories
    2. Images that contain faces in secondary directories
    3. Images that were captured 2 before or two after images with faces in them in time
    4. Randomly chosen, unselected images (no duplicates)

    Images that cannot be opened with PIL (corrupted images) will not be selected.

    :param primary_directory: List of directories as strings.
    :param secondary_directory: List of directories as strings.
    :param target_num_images: The number of images to select.
    :return: The paths to the selected images along with some other info about where the
    images came from.
    """

    sorted_within_directories: List[List[PathAndBoundingBoxes]] = list(
        map(
            _sort_images_by_filename,
            _scan_images_in_directories(
                directories=itertools.chain.from_iterable(
                    [
                        [Path(path_str) for path_str in dir_type]
                        for dir_type in (primary_directory, secondary_directory)
                    ]
                )
            ),
        )
    )

    flattened = list(itertools.chain.from_iterable(sorted_within_directories))

    images_with_faces_for_output, images_with_faces_for_filtering = itertools.tee(
        filter(_contains_face, flattened), 2
    )

    images_with_faces = GenerateWithCount(images_with_faces_for_output)

    surrounding_for_output, surrounding_for_random = itertools.tee(
        set(
            filter(
                lambda k: k is not None,
                itertools.chain.from_iterable(
                    [
                        _images_around_faces(
                            all_possible=images_in_dir,
                            images_with_faces=set(images_with_faces_for_filtering),
                        )
                        for images_in_dir in sorted_within_directories
                    ]
                ),
            )
        ),
        2,
    )

    images_surrounding_images_with_faces = GenerateWithCount(surrounding_for_output)

    def _unselected_random_images() -> Iterator[PathAndBoundingBoxes]:
        """
        Helper function.
        :return: An iterator of images that don't have faces, and don't surround images with faces.
        """
        all_surrounding_images = set(  # Don't actually compute this unless we need it
            surrounding_for_random
        )
        yield from filter(
            lambda p: (not _contains_face(p)) and p not in all_surrounding_images,
            sorted(flattened, key=lambda k: random.random()),
        )

    randomized_images_without_faces = GenerateWithCount(iterator=_unselected_random_images())

    return ImageSelectionOutput(
        path_and_bounding_boxes=list(
            itertools.islice(
                itertools.chain(
                    images_with_faces,
                    images_surrounding_images_with_faces,
                    randomized_images_without_faces,
                ),
                target_num_images,
            )
        ),
        num_img_w_faces=images_with_faces.count,
        num_img_around_img_w_faces=images_surrounding_images_with_faces.count,
        num_randomized_img_wout_faces=randomized_images_without_faces.count,
    )
