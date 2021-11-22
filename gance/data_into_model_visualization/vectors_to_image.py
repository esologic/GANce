"""
Functions around visualizing sets of vectors, resulting in images.
"""
import itertools
from contextlib import _GeneratorContextManager  # pylint: disable=unused-import
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from cv2 import cv2
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing_extensions import Protocol

from gance.apply_spectrogram import compute_spectrogram, reshape_spectrogram_to_vectors
from gance.data_into_model_visualization.vectors_3d import plot_vectors_3d
from gance.data_into_model_visualization.visualization_common import (
    STANDARD_MATPLOTLIB_DPI,
    STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE,
    render_current_matplotlib_frame,
)
from gance.gance_types import RGBInt8ImageType
from gance.vector_sources.vector_sources_common import is_vector, sub_vectors
from gance.vector_sources.vector_types import (
    MatricesLabel,
    SingleMatrix,
    SingleVector,
    VectorsLabel,
)
from gance.video_common import create_video_writer


def multi_plot_vectors(
    vectors_labels: List[VectorsLabel], output_path: Optional[Path] = None
) -> None:
    """
    Plot a multiple vector arrays in 3d on the same figure for comparison.
    :param vectors_labels: A list of the datas to plot and their titles.
    :param output_path: If given, the visualization will be written to this file.
    :return: None
    """

    plots_height = 1
    plots_width = len(vectors_labels)

    fig = plt.figure(figsize=(10 * plots_width, 10))

    axes_data: List[Tuple[Axes, VectorsLabel]] = [
        (fig.add_subplot(plots_height, plots_width, index + 1, projection="3d"), vectors_label)
        for index, vectors_label in enumerate(vectors_labels)
    ]

    for ax, vectors_label in axes_data:
        plot_vectors_3d(ax_3d=ax, vectors_label=vectors_label)

    if output_path is None:
        plt.show()
    else:
        fig.savefig(str(output_path))


def visualize_data_with_spectrogram_and_3d_vectors(
    vectors_label: VectorsLabel,
    inline_spectrogram: bool = True,
    output_path: Optional[Path] = None,
) -> None:
    """
    Create a plot that shows:
    * The data as a single time series plot.
    * The data converted into a spectrogram.
    * The spectrogram converted into a 3d vectors plot.
    :param vectors_label: Holds and describes the vectors to plot.
    :param inline_spectrogram: If true, assumes the input is a time series wave form of some
    kind and computes a spectrogram before visualizing with 3d/spectrogram.
    :param output_path: If given, the visualization will be written to this file.
    :return: None
    """

    fig = plt.figure(figsize=(14, 10))

    fig.suptitle(vectors_label.label)

    axis_3d = fig.add_subplot(2, 2, 1, projection="3d")
    axis_spectrogram = fig.add_subplot(2, 2, 2)
    axis_scatter = fig.add_subplot(2, 1, 2)

    if inline_spectrogram:
        spec_data = compute_spectrogram(vectors_label.data, vectors_label.vector_length)
        data_3d = reshape_spectrogram_to_vectors(
            spectrogram_data=spec_data, vector_length=vectors_label.vector_length
        )
    else:
        spec_data = sub_vectors(vectors_label.data, vector_length=vectors_label.vector_length)
        data_3d = vectors_label.data

    # Draw the spectrogram reshaped in 3D
    plot_vectors_3d(
        axis_3d,
        vectors_label=VectorsLabel(
            data=data_3d,
            vector_length=vectors_label.vector_length,
            label=f"{vectors_label.label} in 3D",
        ),
    )

    # Draw the classical spectrogram
    axis_spectrogram.imshow(
        spec_data,
        origin="lower",
        cmap="viridis",
    )
    axis_spectrogram.set_title("Spectrogram" if inline_spectrogram else "Heatmap")
    axis_spectrogram.axis("tight")
    axis_spectrogram.set_ylabel("Frequency Bin" if inline_spectrogram else "Vector Number")
    axis_spectrogram.set_xlabel("Vector Number" if inline_spectrogram else "Sample # In Vector")

    # Draw the time series waveform
    axis_scatter.plot(vectors_label.data)
    axis_scatter.set_title("Time Series")
    axis_scatter.axis("tight")
    axis_scatter.set_ylabel("Signal Amplitude")
    axis_scatter.set_xlabel("Sample Number")

    # Draw vertical lines on the time series plot to show which parts of the data result in each
    # frame.
    axis_scatter.vlines(
        x=[
            index
            for index in range(len(vectors_label.data))
            if index % vectors_label.vector_length == 0
        ],
        ymin=min(vectors_label.data),
        ymax=max(vectors_label.data),
        colors="red",
        ls=":",
    )

    if output_path is None:
        plt.show()
    else:
        fig.savefig(str(output_path))


class SingleVectorViz(Protocol):
    """
    Defines the function that goes from the x,y values of a vector/matrix to the resulting image
    of that vector.
    """

    def __call__(
        self,
        x_values: np.ndarray,
        y_values: Union[SingleVector, SingleMatrix],
        new_title: Optional[str] = None,
    ) -> "_GeneratorContextManager[np.ndarray]":
        """
        :param x_values: x points
        :param y_values: y points -- This is okay to be of shape (Any, Any)! So you can put a
        matrix here.
        :param new_title: new title for the plot if desired. If not given the init title will
        persist.
        :return: The visualization rendered to an image.
        """


def vector_visualizer(
    y_min: float, y_max: float, title: str, output_width: int, output_height: int
) -> SingleVectorViz:
    """
    Exposes a context manager that allows you to create a matplotlib visualization of x,y
    points. Initial call passes in one time setup params.
    :param y_min: Min y value.
    :param y_max: Max y value.
    :param title: Title of the figure.
    :param output_width: Width of output frame in pixels.
    :param output_height: Height of output frame in pixels.
    :return: Context manager function to actually create the visualization frames.
    """

    # Needs to be this aspect ratio, would be easy to pass these in if needed later on.
    fig = plt.figure(
        figsize=(STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE, STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE),
        dpi=STANDARD_MATPLOTLIB_DPI,
        constrained_layout=False,  # Lets us use `.tight_layout()` later.
    )

    axis = fig.add_subplot(1, 1, 1)
    axis.set_ylim([y_min - 1, y_max + 1])
    axis.set_title(title)

    @contextmanager
    def make_visualization(
        x_values: np.ndarray, y_values: np.ndarray, new_title: Optional[str] = None
    ) -> Iterator[RGBInt8ImageType]:
        """
        Creates the image for the given x,y.
        :param x_values: See protocol.
        :param y_values: See protocol.
        :param new_title: See protocol.
        :return: See protocol.
        """

        if new_title is not None:
            axis.set_title(new_title)

        plotting_y = [y_values] if is_vector(y_values) else y_values

        lines = [
            axis.scatter(x_values, sub_vector, color=color)
            for sub_vector, color in zip(
                plotting_y,
                itertools.cycle(
                    # This is a deterministic way to get the same sequence of colors frame after
                    # frame. There might be a cleaner way to do this.
                    list(mcolors.BASE_COLORS.keys())
                    + list(mcolors.TABLEAU_COLORS.keys())
                ),
            )
        ]

        yield render_current_matplotlib_frame(fig=fig, resolution=(output_width, output_height))

        # Prevents the line from getting drawn more than once.
        for line in lines:
            line.remove()

    return make_visualization


def vectors_to_video(
    labeled_data: Union[VectorsLabel, MatricesLabel],
    output_path: Path,
    video_height: int,
    video_fps: float,
) -> Path:
    """
    Create a canonical video visualization of some vectors/matrices.
    :param labeled_data: To plot.
    :param output_path: Path to video on disk.
    :param video_height: Output height.
    :param video_fps: Output FPS, will effect playback speed.
    :return: The output path once the video has been written.
    """

    make_visualization = vector_visualizer(
        y_min=labeled_data.data.min(),
        y_max=labeled_data.data.max(),
        title=labeled_data.label,
        output_width=video_height,
        output_height=video_height,
    )
    x_values = np.arange(labeled_data.vector_length)

    video = create_video_writer(
        video_path=output_path,
        num_squares=1,
        video_fps=video_fps,
        video_height=video_height,
    )

    for vector in sub_vectors(data=labeled_data.data, vector_length=labeled_data.vector_length):
        with make_visualization(x_values=x_values, y_values=vector) as visualization:
            video.write(cv2.cvtColor(visualization.astype(np.uint8), cv2.COLOR_BGR2RGB))

    video.release()

    return output_path
