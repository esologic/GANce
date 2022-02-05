"""
Functionality related to creating an overlay that tracks eye-containing regions of the foreground
to be placed over the background.
"""

import itertools
from typing import Iterator, List, NamedTuple, Optional, Tuple

import imagehash
from lz.transposition import transpose
from PIL import Image

from gance import faces
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.logger_common import LOGGER
from gance.overlay.overlay_common import (
    BoundingBox,
    OverlayResult,
    bounding_box_distance,
    landmarks_to_bounding_boxes,
)
from gance.overlay.overlay_visualization import OverlayContext


class _FrameOverlayResult(NamedTuple):
    """
    Represents the overlay computation for each frame in the input.
    """

    # If an overlay should be created, the regions of the foreground image that should
    # be written over the background is described by these bounding boxes.
    foreground_bounding_boxes: Optional[List[BoundingBox]] = None

    # Information describing the decision to overlay or not. Consumed by visualization.
    context: Optional[OverlayContext] = OverlayContext()


def compute_eye_tracking_overlay(
    foreground_images: ImageSourceType,
    background_images: ImageSourceType,
    min_phash_distance: int,
    min_bbox_distance: float,
    skip_mask: Optional[List[bool]] = None,
) -> OverlayResult:
    """
    Yield iterators that describe a given overlay operation.
    Attempts to track the eyes found in the foreground image, and then paste them over the same
    position in the background image. Attempts to only do this overlay when the foreground and
    background images are visually similar.
    :param foreground_images: To be displayed on top of background images.
    :param background_images: To be displayed under foreground images.
    :param min_phash_distance: Minimum perceptual hash distance between foreground and background
    for the overlay to be written.
    :param min_bbox_distance: Minimum distance between origins of eye bounding boxes between images
    in foreground and background for the overlay to be written.
    :param skip_mask: List of flags, if the flag is `True`, it's corresponding frame in
    `foreground_images` and `background_images` will not be computed to look for an overlay.
    :return: Series of NTs describing the operation, as well as containing the result.
    """

    face_finder = faces.FaceFinderProxy()

    frame_count = itertools.count()

    def overlay_per_frame(
        packed: Tuple[RGBInt8ImageType, RGBInt8ImageType, bool]
    ) -> _FrameOverlayResult:
        """
        Create the NT describing the overlay operation for a given frame.
        :param packed: input args as a tuple. Composed of:
            foreground_image: Image on top of `background_image`
            background_image: Image under `foreground_image`.
            skip
        :return: NT.
        """

        foreground_image, background_image, skip = packed

        current_frame_number = next(frame_count)

        if skip:
            LOGGER.info(f"Skipping eye tracking overlay for frame #{current_frame_number}")
            return _FrameOverlayResult()

        foreground_bounding_boxes = landmarks_to_bounding_boxes(
            face_finder.face_landmarks(face_image=foreground_image)
        )

        bbox_dist = bounding_box_distance(
            a_boxes=foreground_bounding_boxes,
            b_boxes=landmarks_to_bounding_boxes(
                face_finder.face_landmarks(face_image=background_image)
            ),
        )

        phash_dist = abs(
            imagehash.phash(Image.fromarray(foreground_image))
            - imagehash.phash(Image.fromarray(background_image))
        )

        overlay_flag = phash_dist <= min_phash_distance and (
            bbox_dist < min_bbox_distance if bbox_dist else False
        )

        LOGGER.info(
            f"Computed eye tracking overlay for "
            f"frame #{current_frame_number}, "
            f"content? {overlay_flag}"
        )

        return _FrameOverlayResult(
            foreground_bounding_boxes=foreground_bounding_boxes if overlay_flag else None,
            context=OverlayContext(
                perceptual_hash_distance=phash_dist,
                bbox_distance=bbox_dist,
                overlay_written=overlay_flag,
            ),
        )

    per_frame_results: Iterator[_FrameOverlayResult] = map(
        overlay_per_frame,
        zip(
            foreground_images,
            background_images,
            (
                skip_mask
                if skip_mask is not None
                # This will create an chain of `False` that is the length of the input images.
                else itertools.cycle([False])
            ),
        ),
    )

    # Split the different members from the per-frame tuples into iterables by type.
    # Gives consumer option to totally ignore parts of the result.
    return OverlayResult(*transpose(per_frame_results))
