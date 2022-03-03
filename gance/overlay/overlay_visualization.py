"""
Functionality for visualizing overlay computations. Probably the best way to understand what
is going on during an overlay computation.
"""

import itertools
from typing import Iterator, List, NamedTuple, Optional, Tuple, cast

import more_itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from gance.data_into_network_visualization import visualization_common
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
    image_perceptual_hash_distance: Optional[float] = None

    # How visually similar the bounding box regions of the foreground and background are.
    bbox_perceptual_hash_distance: Optional[float] = None

    # For eye-tracking, how close the eye bounding boxes in the fore/background images
    # are.
    bbox_distance: Optional[float] = None


class VisualizeOverlayThresholds(NamedTuple):
    """
    Set of threshold values to label on an overlay.
    """

    phash_line: float
    bbox_distance_line: float


class YValues(NamedTuple):
    """
    Defines a scatter plot on an axes.
    """

    values: List[float]
    color: str
    label: str


def _setup_axis(
    axis: Axes,
    x_values: np.ndarray,
    y_values: List[YValues],
    title: str,
    horizontal_line_location: Optional[float],
    visualize_all_points: bool,
    y_label: str = "Values",
    x_label: str = "Frame #",
) -> Tuple[float, float]:
    """
    Helper function to set up axes for plotting.
    :param axis: To configure.
    :param x_values: X values of sub-scatter plots.
    :param y_values: List of y values, and data about the values to plot.
    :param title: Axes title.
    :param horizontal_line_location: If given, a horizontal line will be drawn at this location.
    :param visualize_all_points: If given, y axis will be set to show all points on each subplot.
    If false, only +/- 2 standard deviations from the mean will be visualized.
    :param y_label: Y label of axes.
    :param x_label: X label of axes.
    :return: Tuple, (min, max) values given in `y_values`.
    """

    for values in y_values:
        axis.scatter(
            x_values,
            values.values,
            color=values.color,
            label=values.label,
        )

    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.set_xlabel(x_label)
    axis.grid()
    axis.legend(loc="upper right")

    all_y_values = list(
        filter(None, itertools.chain.from_iterable(values.values for values in y_values))
    )

    if all_y_values:
        if visualize_all_points:
            axis_min = min(all_y_values) - 5
            axis_max = max(all_y_values) + 5
        else:
            mean = np.mean(all_y_values)
            std = np.std(all_y_values)
            axis_min = mean - 2 * std
            axis_max = mean + 2 * std
    else:
        axis_min = -5
        axis_max = 5

    axis.set_ylim(axis_min, axis_max)

    axis.hlines(
        y=horizontal_line_location,
        xmin=min(x_values) - 5,
        xmax=max(x_values) + 5,
        linestyles="dotted",
        color="purple",
    )

    return axis_min, axis_max


def visualize_overlay_computation(  # pylint: disable=too-many-locals
    overlay: Iterator[OverlayContext],
    frames_per_context: int,
    video_square_side_length: Optional[int],
    horizontal_lines: Optional[VisualizeOverlayThresholds] = None,
    visualize_all_points: bool = True,
) -> ImageSourceType:
    """
    Consumes the contexts from an overlay computation and produces a visualization of the
    component parts.
    :param overlay: To write visualize.
    :param frames_per_context: The number of adjacent frames to visualize in each frame.
    :param video_square_side_length: Video is composed of a 3x2 grid of square sub-videos,
    each with a side length of this many pixels.
    :param horizontal_lines: Labeled lines to help understand computations.
    :param visualize_all_points: If True, the y axis of each subplot will be stretched such that
    all points in the underlying will be visualized. If false, the y axis will be set to +/- 2
    standard deviations from the mean.
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
            bbox_perceptual_hash_distances,
            image_perceptual_hash_distances,
            bounding_box_distances,
        ) = zip(*current)

        num_frames = len(flags)
        x_axis = np.arange(num_frames)

        hash_axis_min, hash_axis_max = _setup_axis(
            axis=hash_axis,
            x_values=x_axis,
            y_values=[
                YValues(
                    values=cast(List[float], bbox_perceptual_hash_distances),
                    color="red",
                    label="Bounding Boxes",
                ),
                YValues(
                    values=cast(List[float], image_perceptual_hash_distances),
                    color="blue",
                    label="Complete Image",
                ),
            ],
            title="Overlay Discriminator (Image Hashing)",
            horizontal_line_location=horizontal_lines.phash_line if horizontal_lines else None,
            visualize_all_points=visualize_all_points,
        )

        bbox_axis_min, bbox_axis_max = _setup_axis(
            axis=bbox_distance_axis,
            x_values=x_axis,
            y_values=[
                YValues(
                    values=cast(List[float], bounding_box_distances),
                    color="green",
                    label="Bounding Box Distance",
                )
            ],
            title="Overlay Discriminator (Face Tracking)",
            horizontal_line_location=horizontal_lines.bbox_distance_line
            if horizontal_lines
            else None,
            visualize_all_points=visualize_all_points,
        )

        plt.tight_layout()

        video_half_resolution = (video_square_side_length, video_square_side_length)

        for inter_group_index, flag in enumerate(flags):

            LOGGER.info(f"Visualizing overlay for frame #{next(frame_count)}")

            line_color = "green" if flag else "red"

            hash_line = hash_axis.vlines(
                x=inter_group_index, ymin=hash_axis_min, ymax=hash_axis_max, color=line_color
            )

            bbox_line = bbox_distance_axis.vlines(
                x=inter_group_index, ymin=bbox_axis_min, ymax=bbox_axis_max, color=line_color
            )

            yield visualization_common.render_current_matplotlib_frame(
                fig=fig, resolution=video_half_resolution
            )

            hash_line.remove()
            bbox_line.remove()

        for axes in fig.axes:
            axes.clear()
