"""
Write parts of one video on top of another.
Do things like track eyes to interesting effect.
"""

import itertools
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance

from gance.gance_types import RGBInt8ImageType
from gance.image_sources.image_sources_common import ImageResolution, image_resolution
from gance.overlay.overlay_visualization import OverlayContext


class BoundingBox(NamedTuple):
    """
    Describes a bounding box rectangle as output by opencv.
    """

    x: int
    y: int
    width: int
    height: int


def convert_to_pil_box(bounding_box: BoundingBox) -> Tuple[int, int, int, int]:
    """
    PIL's crop function takes their bounding boxes in a different order. This function
    converts the canonical representation to that order.
    :param bounding_box: To convert.
    :return: (left, upper, right, lower) -- This is the order PIL wants.
    """

    return (
        bounding_box.x,
        bounding_box.y,
        bounding_box.x + bounding_box.width,
        bounding_box.y + bounding_box.height,
    )


def landmarks_to_bounding_boxes(landmarks: List[Dict[str, Tuple[int, ...]]]) -> List[BoundingBox]:
    """
    For each of the sets of keypoints, pull out the interesting ones, and draw a bounding box
    around them. Currently, the left and right eye keypoints are used.
    :param landmarks: From the face recognition library.
    :return: List of bounding boxes.
    """

    return [
        BoundingBox(*cv2.boundingRect(np.array(landmark["left_eye"] + landmark["right_eye"])))
        for landmark in landmarks
    ]


def bounding_box_center(bounding_box: BoundingBox) -> Tuple[float, float]:
    """
    Finds the center x,y coordinate of a given bounding box.
    :param bounding_box: Box to analyze.
    :return: (x,y) of center.
    """

    return (bounding_box.x + bounding_box.width / 2), (bounding_box.y + bounding_box.height / 2)


class DistanceBoxes(NamedTuple):
    """
    Intermediate type.
    Stores the distance in pixels between the centers of the two bounding boxes.
    Allows the boxes themselves to be maintained for further use.
    """

    # Distance between the boxes in pixels.
    distance: float

    a_box: BoundingBox
    b_box: BoundingBox


def bounding_box_distance(
    a_boxes: List[BoundingBox], b_boxes: List[BoundingBox]
) -> Optional[DistanceBoxes]:
    """
    Calculate the minimum distance between two sets of bounding boxes.
    :param a_boxes: Left side.
    :param b_boxes: Right side.
    :return: Minimum distance between the centers of these boxes in pixels.
    """
    return min(
        [
            DistanceBoxes(
                distance=float(
                    distance.euclidean(bounding_box_center(a_box), bounding_box_center(b_box))
                ),
                a_box=a_box,
                b_box=b_box,
            )
            for a_box, b_box in itertools.product(a_boxes, b_boxes)
        ],
        key=lambda distance_box: distance_box.distance,
        default=None,
    )


def _draw_mask(  # pylint: disable=too-many-locals
    resolution: ImageResolution, bounding_boxes: List[BoundingBox]
) -> "Image":
    """
    Draw bounding boxes as a white mask. Edges of bounding boxes are
    included in mask.
    :param resolution: The size of the resulting image.
    :param bounding_boxes: To draw as a mask on the image.
    :return: The input image, but now with the bounding boxes down onto it.
    """

    output = Image.new("L", tuple(resolution))
    draw = ImageDraw.Draw(output)

    for bounding_box in bounding_boxes:

        x, y, w, h = bounding_box

        # These pads need to scale with the input size.
        # Eventually would like to pass these kinds of magic numbers in.
        y_pad = resolution.width * 0.058
        x_pad = resolution.height * 0.098

        y_center = y + (h / 2)
        y_lower = y_center + y_pad
        y_upper = y_center - y_pad
        x_left = x - x_pad
        x_right = x + (w + x_pad)

        draw.polygon(
            [(x_left, y_lower), (x_right, y_lower), (x_right, y_upper), (x_left, y_upper)],
            outline=255,
            fill=255,
        )

    return output


def _apply_mask(
    foreground_image: "Image", background_image: RGBInt8ImageType, mask: "Image"
) -> RGBInt8ImageType:
    """
    Writes a masked region of the foreground onto the background.
    :param foreground_image: An image that has already been converted to a PIL Image. This will
    be drawn onto the background.
    :param background_image: The region of `foreground_image` will be drawn onto this image.
    :param mask: Describes the region of `foreground_image` to draw onto `background_image`.
    :return: The new image in the canonical type.
    """
    return cast(
        RGBInt8ImageType,
        np.asarray(Image.composite(foreground_image, Image.fromarray(background_image), mask)),
    )


def write_boxes_onto_image(
    foreground_image: RGBInt8ImageType,
    background_image: RGBInt8ImageType,
    bounding_boxes: List[BoundingBox],
) -> RGBInt8ImageType:
    """
    Write regions of the foreground image onto the background image.
    Return the result as a new image.
    :param foreground_image: To be drawn onto the background.
    :param background_image: The be drawn under the foreground.
    :param bounding_boxes: The regions of the foreground to be drawn onto the background.
    :return: A new image, with the foreground regions written onto the background.
    """

    foreground = Image.fromarray(foreground_image)

    return _apply_mask(
        foreground_image=foreground,
        background_image=background_image,
        mask=_draw_mask(image_resolution(foreground_image), bounding_boxes),
    )


class OverlayResult(NamedTuple):
    """
    The different output streams from an eye tracking overlay computation.
    See the docs for `_FrameOverlayResult` for meaning as to what the different
    members are here, these are iterators of those types.
    Note: really important that the order of the members matches `_FrameOverlayResult`.
    """

    bbox_lists: Iterator[Optional[List[BoundingBox]]]
    contexts: Iterator[OverlayContext]
