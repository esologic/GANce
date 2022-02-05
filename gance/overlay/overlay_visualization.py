"""
Functionality for visualizing overlay computations. Probably the best way to understand what
is going on during an overlay computation.
"""

import itertools
from typing import Iterator, NamedTuple, Optional

import more_itertools
import numpy as np
from matplotlib import pyplot as plt

from gance.data_into_model_visualization import visualization_common
from gance.gance_types import ImageSourceType
from gance.logger_common import LOGGER


class OverlayContext(NamedTuple):
    """
    Component parts of an overlay computation that lead to the decision to write the overlay
    or not.
    """

    # If the overlay should be written
    overlay_written: bool = False

    # The following are params considered when computing the overlay.

    # How visually similar the foreground and background are.
    perceptual_hash_distance: Optional[float] = None

    # For eye-tracking, how close the eye bounding boxes in the fore/background images
    # are.
    bbox_distance: Optional[float] = None


def visualize_overlay_computation(  # pylint: disable=too-many-locals
    overlay: Iterator[OverlayContext],
    frames_per_context: int,
    video_square_side_length: Optional[int],
) -> ImageSourceType:
    """
    Consumes the contexts from an overlay computation and produces a visualization of the
    component parts.
    :param overlay: To write visualize.
    :param frames_per_context: The number of adjacent frames to visualize in each frame.
    :param video_square_side_length: Video is composed of a 3x2 grid of square sub-videos,
    each with a side length of this many pixels.
    :return: The frames of the visualization.
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
