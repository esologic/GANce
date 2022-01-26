"""
Write parts of one video on top of another.
Do things like track eyes to interesting effect.
"""

import itertools
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, cast

import imagehash
import more_itertools
import numpy as np
from cv2 import cv2
from lz.transposition import transpose
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from scipy.spatial import distance

from gance import faces
from gance.data_into_model_visualization import visualization_common
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.logger_common import LOGGER


class BoundingBoxType(NamedTuple):
    """
    Describes a bounding box rectangle as output by opencv.
    """

    x: int
    y: int
    width: int
    height: int


class OverlayContext(NamedTuple):
    """
    TODO -- fill this out
    """

    overlay_written: bool

    # The following are params considered when computing the overlay.
    perceptual_hash_distance: Optional[float] = None
    average_hash_distance: Optional[float] = None
    difference_hash_distance: Optional[float] = None
    wavelet_hash_distance: Optional[float] = None
    hashes_average_distance: Optional[float] = None
    bbox_distance: Optional[float] = None


def landmarks_to_bounding_boxes(
    landmarks: List[Dict[str, Tuple[int, ...]]]
) -> List[BoundingBoxType]:
    """
    For each of the sets of keypoints, pull out the interesting ones, and draw a bounding box
    around them. Currently, the left and right eye keypoints are used.
    :param landmarks: From the face recognition library.
    :return: List of bounding boxes.
    """

    return [
        BoundingBoxType(*cv2.boundingRect(np.array(landmark["left_eye"] + landmark["right_eye"])))
        for landmark in landmarks
    ]


def bounding_box_center(bounding_box: BoundingBoxType) -> Tuple[float, float]:
    """
    Finds the center x,y coordinate of a given bounding box.
    :param bounding_box: Box to analyze.
    :return: (x,y) of center.
    """

    return (bounding_box.x + bounding_box.width / 2), (bounding_box.y + bounding_box.height / 2)


def bounding_box_distance(
    a_boxes: List[BoundingBoxType], b_boxes: List[BoundingBoxType]
) -> Optional[float]:
    """
    Calculate the minimum distance between two sets of bounding boxes.
    :param a_boxes: Left side.
    :param b_boxes: Right side.
    :return: Minimum distance between the origins of these boxes in pixels.
    """

    if a_boxes and b_boxes:
        # Only take the X,Y points here
        a_origin = [bounding_box_center(box) for box in a_boxes]
        b_origin = [bounding_box_center(box) for box in b_boxes]

        return min(
            [
                float(distance.euclidean(target_point, final_point))
                for target_point, final_point in itertools.product(a_origin, b_origin)
            ]
        )
    else:
        return None


def draw_mask(destination: "Image", bounding_boxes: List[BoundingBoxType]) -> Image:
    """
    Draw bounding boxes as a white mask on a given image. Edges of bounding boxes are
    included in mask.
    :param destination: The image to draw the boxes on.
    :param bounding_boxes: To draw as a mask on the image.
    :return: The input image, but now with the bounding boxes down onto it.
    """

    draw = ImageDraw.Draw(destination)

    for bounding_box in bounding_boxes:

        x, y, w, h = bounding_box

        y_center = y + (h / 2)
        y_lower = y_center + 60
        y_upper = y_center - 60
        x_left = x - 200
        x_right = x + (w + 200)

        draw.polygon(
            [(x_left, y_lower), (x_right, y_lower), (x_right, y_upper), (x_left, y_upper)],
            outline=255,
            fill=255,
        )

    return destination


def apply_mask(
    foreground_image: RGBInt8ImageType, background_image: RGBInt8ImageType, mask: "Image"
) -> RGBInt8ImageType:
    """

    :param foreground_image:
    :param background_image:
    :param mask:
    :return:
    """
    return cast(
        RGBInt8ImageType,
        np.asarray(
            Image.composite(
                Image.fromarray(foreground_image), Image.fromarray(background_image), mask
            )
        ),
    )


class EyeTrackingOverlay(NamedTuple):
    """
    The different output streams from an eye tracking overlay computation.
    See the docs for `_FrameOverlayResult` for meaning as to what the different
    members are here, these are iterators of those types.
    Note: really important that the order of the members matches `_FrameOverlayResult`.
    """

    masks: Iterator["Image"]
    foregrounds: ImageSourceType
    backgrounds: ImageSourceType
    contexts: Iterator[OverlayContext]


class _FrameOverlayResult(NamedTuple):
    """
    Represents the overlay computation for each frame in the input.
    """

    # A PIL image that is a mask of the part of `foreground` that should be overlayed onto
    # `background`.
    # Note that this mask can be 'empty', and there might be no part of `foreground` that should
    # be added to `background`.
    mask: "Image"

    # The image that will be partially drawn onto `background` as defined by `mask`.
    foreground: RGBInt8ImageType

    # The image that the mask will be drawn onto.
    background: RGBInt8ImageType

    # Information describing the decision to overlay or not. Consumed by visualization.
    context: OverlayContext


def compute_eye_tracking_overlay(
    foreground_images: ImageSourceType,
    background_images: ImageSourceType,
    min_phash_distance: int = 30,
    min_bbox_distance: float = 50.0,
) -> EyeTrackingOverlay:
    """
    Yield iterators that describe a given overlay operation.
    Attempts to track the eyes found in the foreground image, and then paste them over the same
    position in the background image. Attempts to only do this overlay when the foreground and
    background images are visually similar.
    :param foreground_images: To be displayed on top of background images.
    :param background_images: To be displayed under foreground images.
    :return: Series of NTs describing the operation, as well as containing the result.
    """

    face_finder = faces.FaceFinderProxy()

    def overlay_per_frame(
        packed: Tuple[int, Tuple[RGBInt8ImageType, RGBInt8ImageType]]
    ) -> _FrameOverlayResult:
        """
        Create the NT describing the overlay operation for a given frame.
        :param packed: input args as a tuple. Composed of:
            index: Frame index (for logging)
            foreground_image: Image on top of `background_image`
            background_image: Image under `foreground_image`.
        :return: NT.
        """

        (index, (foreground_image, background_image)) = packed

        LOGGER.info(f"Computing eye tracking overlay for frame #{index}")

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

        blank_mask = Image.new("L", foreground_image.shape[0:2], 0)

        mask = (
            draw_mask(destination=blank_mask, bounding_boxes=foreground_bounding_boxes)
            if overlay_flag
            else blank_mask
        )

        return _FrameOverlayResult(
            mask=mask,
            context=OverlayContext(
                perceptual_hash_distance=phash_dist,
                bbox_distance=bbox_dist,
                overlay_written=overlay_flag,
            ),
            foreground=foreground_image,
            background=background_image,
        )

    per_frame_results: Iterator[_FrameOverlayResult] = map(
        overlay_per_frame, enumerate(zip(foreground_images, background_images))
    )

    masks, foregrounds, backgrounds, contexts = transpose(per_frame_results)

    # Split the different members from the per-frame tuples into iterables by type.
    # Gives consumer option to totally ignore parts of the result.
    return EyeTrackingOverlay(
        masks=masks, foregrounds=foregrounds, backgrounds=backgrounds, contexts=contexts
    )


def visualize_overlay_computation(  # pylint: disable=too-many-locals
    overlay: Iterator[OverlayContext],
    frames_per_context: int,
    video_square_side_length: Optional[int],
) -> ImageSourceType:
    """
    Writes an overlay iterator to disk.
    :param overlay: To write.
    :param video_path: Path to write the video to on disk.
    :param video_square_side_length: Video is composed of a 3x2 grid of square sub-videos,
    each with a side length of this many pixels.
    :return: None
    """

    fig = visualization_common.standard_matplotlib_figure()

    hash_axis, bbox_distance_axis = fig.subplots(nrows=2, ncols=1)

    for group_index, group_of_frames in enumerate(
        more_itertools.grouper(overlay, frames_per_context)
    ):

        current: Iterator[OverlayContext] = filter(None, group_of_frames)

        (
            flags,
            perceptual_hash_distances,
            average_hash_distances,
            difference_hash_distances,
            wavelet_hash_distance,
            hashes_average_distance,
            bounding_box_distances,
        ) = zip(*current)

        num_frames = len(flags)
        x_axis = np.arange(num_frames)

        hash_axis.scatter(
            x_axis,
            perceptual_hash_distances,
            color="red",
            label="P Hash Distance",
            marker="x",
            alpha=0.5,
        )

        hash_axis.scatter(
            x_axis,
            average_hash_distances,
            color="purple",
            label="A Hash Distance",
            marker="x",
            alpha=0.5,
        )

        hash_axis.scatter(
            x_axis,
            difference_hash_distances,
            color="blue",
            label="D Hash Distance",
            marker="x",
            alpha=0.5,
        )

        hash_axis.scatter(
            x_axis,
            wavelet_hash_distance,
            color="brown",
            label="W Hash Distance",
            marker="x",
            alpha=0.5,
        )

        hash_axis.scatter(
            x_axis, hashes_average_distance, color="green", label="Hashes Average Distance"
        )

        hash_axis.set_title("Overlay Discriminator (Image Hashing)")
        hash_axis.set_ylabel("Values")
        hash_axis.set_xlabel("Frame #")
        hash_axis.grid()
        hash_axis.legend(loc="upper right")

        hash_all_y_values = [value for value in perceptual_hash_distances if value is not None]
        hash_axis_min = min(hash_all_y_values) - 5
        hash_axis_max = max(hash_all_y_values) + 5

        hash_axis.set_ylim(hash_axis_min, hash_axis_max)

        bbox_distance_axis.scatter(
            x_axis, bounding_box_distances, color="green", label="Bounding Box Distance"
        )

        bbox_distance_axis.set_title("Overlay Discriminator (Face Tracking)")
        bbox_distance_axis.set_ylabel("Distance (Pixels)")
        bbox_distance_axis.set_xlabel("Frame #")
        bbox_distance_axis.grid()
        bbox_distance_axis.legend(loc="upper right")

        bbox_all_y_values = [value for value in bounding_box_distances if value is not None]

        if bbox_all_y_values:
            bbox_axis_min = min(bbox_all_y_values) - 5
            bbox_axis_max = max(bbox_all_y_values) + 5
            bbox_distance_axis.set_ylim(bbox_axis_min, bbox_axis_max)

        plt.tight_layout()

        video_half_resolution = (video_square_side_length, video_square_side_length)

        for inter_group_index, flag in enumerate(flags):

            LOGGER.info(f"Visualizing overlay for frame #{group_index + inter_group_index}")

            line_color = "green" if flag else "red"

            hash_line = hash_axis.vlines(
                x=inter_group_index, ymin=hash_axis_min, ymax=hash_axis_max, color=line_color
            )

            if bbox_all_y_values:
                bbox_line = bbox_distance_axis.vlines(
                    x=inter_group_index, ymin=bbox_axis_min, ymax=bbox_axis_max, color=line_color
                )

            yield visualization_common.render_current_matplotlib_frame(
                fig=fig, resolution=video_half_resolution
            )

            hash_line.remove()

            if bbox_all_y_values:
                bbox_line.remove()

        for axes in fig.axes:
            axes.clear()
