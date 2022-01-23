"""
Write parts of one video on top of another.
Do things like track eyes to interesting effect.
"""

from gance.image_sources import video_common
import itertools
import time
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, cast

import imagehash
import more_itertools
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from scipy.spatial import distance

from gance import faces
from gance.assets import OUTPUT_DIRECTORY, PROJECTION_FILE_PATH
from gance.data_into_model_visualization.visualization_common import (
    STANDARD_MATPLOTLIB_DPI,
    render_current_matplotlib_frame,
)
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.image_sources.video_common import create_video_writer
from gance.logger_common import LOGGER
from gance.projection.projection_file_reader import load_projection_file


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


class FrameOverlayResult(NamedTuple):
    """
    Represents the overlay computation for each frame in the input.
    TODO: Going to change, improve docs when settled.
    """

    mask: "Image"
    foreground: RGBInt8ImageType
    background: RGBInt8ImageType
    context: OverlayContext


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
    foreground_image: RGBInt8ImageType, background_image: RGBInt8ImageType, mask: Image
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


def compute_eye_tracking_overlay(  # pylint: disable=too-many-locals
    foreground_images: Iterator[RGBInt8ImageType], background_images: Iterator[RGBInt8ImageType]
) -> Iterator[FrameOverlayResult]:
    """
    Yield a series of NTs that describe a given overlay operation.
    Attempts to track the eyes found in the foreground image, and then paste them over the same
    position in the background image. Attempts to only do this overlay when the foreground and
    background images are visually similar.
    :param foreground_images: To be displayed on top of background images.
    :param background_images: To be displayed under foreground images.
    :return: Series of NTs describing the operation, as well as containing the result.
    """

    face_finder = faces.FaceFinderProxy()

    for index, (foreground_image, background_image) in enumerate(
        zip(foreground_images, background_images)
    ):

        LOGGER.info(f"Computing eye tracking overlay for frame #{index}")

        fore_pil_image = Image.fromarray(foreground_image)
        back_pil_image = Image.fromarray(background_image)

        foreground_bounding_boxes = landmarks_to_bounding_boxes(
            face_finder.face_landmarks(face_image=foreground_image)
        )

        bbox_dist = bounding_box_distance(
            a_boxes=foreground_bounding_boxes,
            b_boxes=landmarks_to_bounding_boxes(
                face_finder.face_landmarks(face_image=background_image)
            ),
        )

        phash_dist = abs(imagehash.phash(fore_pil_image) - imagehash.phash(back_pil_image))

        overlay_flag = phash_dist <= 30 and (bbox_dist < 50 if bbox_dist else False)

        blank_mask = Image.new("L", foreground_image.shape[0:2], 0)

        mask = (
            draw_mask(destination=blank_mask, bounding_boxes=foreground_bounding_boxes)
            if overlay_flag
            else blank_mask
        )

        yield FrameOverlayResult(
            mask=mask,
            context=OverlayContext(
                perceptual_hash_distance=phash_dist,
                bbox_distance=bbox_dist,
                overlay_written=overlay_flag,
            ),
            foreground=foreground_image,
            background=background_image,
        )


def render_overlay(  # pylint: disable=too-many-locals
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

    fig = plt.figure(
        figsize=(1, 1),
        dpi=STANDARD_MATPLOTLIB_DPI,
        constrained_layout=False,  # Lets us use `.tight_layout()` later.
    )

    hash_axis, bbox_distance_axis = fig.subplots(nrows=2, ncols=1)

    for group_index, group_of_frames in enumerate(
        more_itertools.grouper(overlay, frames_per_context)
    ):

        LOGGER.info(f"Working through frame group #{group_index}")

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

        for index, flag in enumerate(flags):

            line_color = "green" if flag else "red"

            hash_line = hash_axis.vlines(
                x=index, ymin=hash_axis_min, ymax=hash_axis_max, color=line_color
            )

            if bbox_all_y_values:
                bbox_line = bbox_distance_axis.vlines(
                    x=index, ymin=bbox_axis_min, ymax=bbox_axis_max, color=line_color
                )

            graph = render_current_matplotlib_frame(fig=fig, resolution=video_half_resolution)

            hash_line.remove()

            if bbox_all_y_values:
                bbox_line.remove()

            LOGGER.info(f"Wrote frame: {index + 1}/{num_frames}")

            yield graph

        for axes in fig.axes:
            axes.clear()


if __name__ == "__main__":

    with load_projection_file(Path(PROJECTION_FILE_PATH)) as reader:

        render_overlay(
            overlay=itertools.islice(
                compute_eye_tracking_overlay(
                    foreground_images=reader.target_images,
                    background_images=reader.final_images,
                ),
                100,
            ),
            video_path=OUTPUT_DIRECTORY.joinpath(f"{int(time.time())}_sample.mp4"),
            video_square_side_length=500,
            frames_per_context=100,
        )

        video_common.write_source_to_disk(
            source=frames,
            video_path=video_path,
            video_fps=video_fps,
        )

