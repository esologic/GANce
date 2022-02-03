"""
Write parts of one video on top of another.
Do things like track eyes to interesting effect.
"""

import itertools
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, cast

import imagehash
import more_itertools
import numpy as np
import pandas as pd
from cv2 import cv2
from lz.transposition import transpose
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from scipy.spatial import distance

from gance import faces
from gance.assets import NOVA_PATH
from gance.data_into_model_visualization import visualization_common
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.image_sources.image_sources_common import ImageResolution, image_resolution
from gance.logger_common import LOGGER
from gance.vector_sources import music, vector_reduction
from gance.vector_sources.vector_reduction import DataLabel, ResultLayers


class BoundingBox(NamedTuple):
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

    overlay_written: bool = False

    # The following are params considered when computing the overlay.
    perceptual_hash_distance: Optional[float] = None
    average_hash_distance: Optional[float] = None
    difference_hash_distance: Optional[float] = None
    wavelet_hash_distance: Optional[float] = None
    hashes_average_distance: Optional[float] = None
    bbox_distance: Optional[float] = None


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


def bounding_box_distance(
    a_boxes: List[BoundingBox], b_boxes: List[BoundingBox]
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


def _draw_mask(resolution: ImageResolution, bounding_boxes: List[BoundingBox]) -> "Image":
    """
    Draw bounding boxes as a white mask on a given image. Edges of bounding boxes are
    included in mask.
    :param destination: The image to draw the boxes on.
    :param bounding_boxes: To draw as a mask on the image.
    :return: The input image, but now with the bounding boxes down onto it.
    """

    output = Image.new("L", tuple(resolution))
    draw = ImageDraw.Draw(output)

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

    return output


def _apply_mask(
    foreground_image: "Image", background_image: RGBInt8ImageType, mask: "Image"
) -> RGBInt8ImageType:
    """
    Writes a masked region of the foreground
    :param foreground_image: An image that has already been converted to a PIL Image
    :param background_image:
    :param mask:
    :return:
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

    :param foreground_image:
    :param background_image:
    :param bounding_boxes:
    :return:
    """

    foreground = Image.fromarray(foreground_image)

    mask = _draw_mask(image_resolution(foreground_image), bounding_boxes)

    output = _apply_mask(
        foreground_image=foreground,
        background_image=background_image,
        mask=mask,
    )

    return output


class EyeTrackingOverlay(NamedTuple):
    """
    The different output streams from an eye tracking overlay computation.
    See the docs for `_FrameOverlayResult` for meaning as to what the different
    members are here, these are iterators of those types.
    Note: really important that the order of the members matches `_FrameOverlayResult`.
    """

    bbox_lists: Iterator[Optional[List[BoundingBox]]]
    contexts: Iterator[OverlayContext]


class _FrameOverlayResult(NamedTuple):
    """
    Represents the overlay computation for each frame in the input.
    """

    # If an overlay should be created, the regions of the foreground image that should
    # be written over the background is described by these bounding boxes.
    foreground_bounding_boxes: Optional[List[BoundingBox]] = None

    # Information describing the decision to overlay or not. Consumed by visualization.
    context: Optional[OverlayContext] = None


def compute_eye_tracking_overlay(
    foreground_images: ImageSourceType,
    background_images: ImageSourceType,
    min_phash_distance: int = 30,
    min_bbox_distance: float = 50.0,
    skip_mask: Optional[List[bool]] = None,
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
    return EyeTrackingOverlay(*transpose(per_frame_results))


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

    frame_count = itertools.count()

    for group_of_frames in more_itertools.grouper(overlay, frames_per_context):

        current: Iterator[OverlayContext] = filter(None, group_of_frames)

        # When we unzip here, the left side of the equation are all lists!
        # So it's okay to iterate over them more than once.
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
        hash_axis_min = min(hash_all_y_values) - 5 if hash_all_y_values else -5
        hash_axis_max = max(hash_all_y_values) + 5 if hash_all_y_values else 5

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

            LOGGER.info(f"Visualizing overlay for frame #{next(frame_count)}")

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


if __name__ == "__main__":

    vector_length = 512
    fps = 30.0

    time_series_audio_vectors = music.read_wav_scale_for_video(
        wav=NOVA_PATH,
        vector_length=vector_length,
        frames_per_second=fps,
    )

    overlay_mask = vector_reduction.rolling_sum_results_layers(
        vector_reduction.absolute_value_results_layers(
            results_layers=ResultLayers(
                result=DataLabel(
                    data=vector_reduction.derive_results_layers(
                        vector_reduction.reduce_vector_gzip_compression_rolling_average(
                            time_series_audio_vectors=time_series_audio_vectors.wav_data,
                            vector_length=vector_length,
                        ),
                        order=1,
                    ).result.data,
                    label="Gzipped audio, smoothed, averaged, 1st order derivation.",
                ),
            ),
        ),
        window_length=10,
    )

    music_overlay_filter_mask: List[bool] = list(
        pd.Series(overlay_mask.result.data).fillna(np.inf) > 20
    )

    print("stop")
