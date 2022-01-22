"""
Write parts of one video on top of another.
Do things like track eyes to interesting effect.
"""

import itertools
import statistics
import time
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, cast

import face_recognition
import imagehash
import more_itertools
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from scipy.spatial import distance

from gance.assets import OUTPUT_DIRECTORY, PROJECTION_FILE_PATH
from gance.data_into_model_visualization.visualization_common import (
    STANDARD_MATPLOTLIB_DPI,
    render_current_matplotlib_frame,
)
from gance.gance_types import RGBInt8ImageType
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


class FrameOverlayResult(NamedTuple):
    """
    Represents the overlay computation for each frame in the input.
    TODO: Going to change, improve docs when settled.
    """

    # Output image, contains the overlaid frame.
    frame: RGBInt8ImageType

    # The following are params considered when computing the overlay.
    perceptual_hash_distance: float
    average_hash_distance: float
    difference_hash_distance: float
    wavelet_hash_distance: float
    hashes_average_distance: float

    bbox_distance: float

    overlay_written: bool


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

    for index, (foreground_image, background_image) in enumerate(
        zip(foreground_images, background_images)
    ):

        LOGGER.info(f"Creating frame #{index}")

        fore_pil_image = Image.fromarray(foreground_image)
        back_pil_image = Image.fromarray(background_image)

        foreground_bounding_boxes = landmarks_to_bounding_boxes(
            face_recognition.face_landmarks(face_image=foreground_image)
        )

        bbox_dist = bounding_box_distance(
            a_boxes=foreground_bounding_boxes,
            b_boxes=landmarks_to_bounding_boxes(
                face_recognition.face_landmarks(face_image=background_image)
            ),
        )

        phash_dist = abs(imagehash.phash(fore_pil_image) - imagehash.phash(back_pil_image))
        ahash_dist = abs(
            imagehash.average_hash(fore_pil_image) - imagehash.average_hash(back_pil_image)
        )
        dhash_dist = abs(imagehash.dhash(fore_pil_image) - imagehash.dhash(back_pil_image))
        whash_dist = abs(imagehash.whash(fore_pil_image) - imagehash.whash(back_pil_image))

        hashes_averages = statistics.mean([phash_dist, ahash_dist, whash_dist])

        overlay_flag = phash_dist <= 30 and (bbox_dist < 50 if bbox_dist else False)

        blank_mask = Image.new("L", foreground_image.shape[0:2], 0)

        mask = (
            draw_mask(destination=blank_mask, bounding_boxes=foreground_bounding_boxes)
            if overlay_flag
            else blank_mask
        )

        yield FrameOverlayResult(
            frame=cv2.hconcat(
                [
                    cast(
                        RGBInt8ImageType,
                        np.asarray(Image.composite(fore_pil_image, back_pil_image, mask)),
                    ),
                    foreground_image,
                    background_image,
                ]
            ),
            perceptual_hash_distance=phash_dist,
            average_hash_distance=ahash_dist,
            difference_hash_distance=dhash_dist,
            wavelet_hash_distance=whash_dist,
            hashes_average_distance=hashes_averages,
            bbox_distance=bbox_dist,
            overlay_written=overlay_flag,
        )


def render_overlay(  # pylint: disable=too-many-locals
    overlay: Iterator[FrameOverlayResult],
    video_path: Path,
    frames_per_context: int,
    video_square_side_length: Optional[int],
) -> None:
    """
    Writes an overlay iterator to disk.
    :param overlay: To write.
    :param video_path: Path to write the video to on disk.
    :param video_square_side_length: Video is composed of a 3x2 grid of square sub-videos,
    each with a side length of this many pixels.
    :return: None
    """

    LOGGER.info(f"Writing Output Video: {video_path}")

    video = create_video_writer(
        video_path=video_path,
        num_squares_width=3,
        num_squares_height=2,
        video_fps=reader.projection_attributes.projection_fps,
        video_height=video_square_side_length,
    )

    fig = plt.figure(
        figsize=(
            4 * 3,
            4 * 1,
        ),
        dpi=STANDARD_MATPLOTLIB_DPI,
        constrained_layout=False,  # Lets us use `.tight_layout()` later.
    )

    hash_axis, bbox_distance_axis = fig.subplots(nrows=2, ncols=1)

    for group_of_frames in more_itertools.grouper(overlay, frames_per_context):

        current = filter(None, group_of_frames)

        (
            frames,
            perceptual_hash_distances,
            average_hash_distances,
            difference_hash_distances,
            wavelet_hash_distance,
            hashes_average_distance,
            bounding_box_distances,
            flags,
        ) = zip(*current)

        num_frames = len(frames)
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

        video_half_resolution = (3 * video_square_side_length, video_square_side_length)

        for index, (video_frame, flag) in enumerate(zip(frames, flags)):

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

            video.write(
                cv2.cvtColor(
                    cv2.vconcat([cv2.resize(video_frame, video_half_resolution), graph]).astype(
                        np.uint8
                    ),
                    cv2.COLOR_BGR2RGB,
                )
            )

            LOGGER.info(f"Wrote frame: {index + 1}/{num_frames}")

        for axes in fig.axes:
            axes.clear()

    video.release()


if __name__ == "__main__":

    with load_projection_file(Path(PROJECTION_FILE_PATH)) as reader:

        render_overlay(
            overlay=itertools.islice(
                compute_eye_tracking_overlay(
                    foreground_images=reader.target_images,
                    background_images=reader.final_images,
                ),
                None,
            ),
            video_path=OUTPUT_DIRECTORY.joinpath(f"{int(time.time())}_sample.mp4"),
            video_square_side_length=500,
            frames_per_context=100,
        )
