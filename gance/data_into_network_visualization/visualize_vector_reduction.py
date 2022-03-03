"""
Functions to visualize the vector reduction process.
"""
import itertools
from pathlib import Path
from typing import Optional

import more_itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from gance.data_into_network_visualization import visualization_common
from gance.data_into_network_visualization.visualization_common import (
    VectorsReducer,
    infinite_colors,
    standard_matplotlib_figure,
)
from gance.gance_types import ImageSourceType
from gance.logger_common import LOGGER
from gance.vector_sources import vector_reduction
from gance.vector_sources.music import read_wavs_scale_for_video
from gance.vector_sources.vector_reduction import ResultLayers


def visualize_reducer_output(audio_path: Path, reducer: VectorsReducer) -> None:
    """
    Demo to visualize RMS values of a known mp3.
    :return: None
    """

    vector_length = 1000

    audio = read_wavs_scale_for_video(
        wavs=[audio_path], vector_length=vector_length, frames_per_second=60.0
    ).wav_data

    reduced = vector_reduction.quantize_results_layers(
        results_layers=reducer(time_series_audio_vectors=audio, vector_length=vector_length),
        network_indices=list(range(30)),
    )

    # Needs to be this aspect ratio
    fig = standard_matplotlib_figure()

    gs = fig.add_gridspec(nrows=4)

    layers_axis = fig.add_subplot(gs[0:3, :])
    result_axis = fig.add_subplot(gs[3:4, :])

    # Plot the individual layers
    layers_axis.set_ylim(
        (
            min([layer.data.min() for layer in reduced.layers]),
            max([layer.data.max() for layer in reduced.layers]),
        )
    )

    width = len(reduced.result.data)
    x_values = np.arange(width)

    for layer, color in zip(
        reduced.layers,
        infinite_colors(),
    ):
        layers_axis.plot(x_values, layer.data, alpha=0.5, label=layer.label, color=str(color))

    layers_axis.legend(loc="upper right")
    layers_axis.set_title("Compositional Layers")
    layers_axis.set_ylabel("Signal Amplitude")
    layers_axis.set_xlabel("Frame #")

    # Plot the result
    result_axis.plot(x_values, reduced.result.data)
    result_axis.set_title("Quantized Result")
    result_axis.set_ylabel("network Index")
    result_axis.set_xlabel("Frame #")

    fig.suptitle(f"network Selection for {audio_path.name}", fontsize=16)

    plt.tight_layout()
    plt.show()


def visualize_result_layers(  # pylint: disable=too-many-locals
    result_layers: ResultLayers,
    video_height: int,
    frames_per_context: int,
    title: str,
    horizontal_line: Optional[float] = None,
) -> ImageSourceType:
    """
    Create a stream of images that visualize a `ResultLayers` object. Useful for understanding
    how the resulting data was created.
    :param result_layers: To visualize.
    :param video_height: Side length in pixels.
    :param frames_per_context: Number of time-series points to display at once.
    :param title: To be written on the top of the figure.
    :param horizontal_line: Arbitrary line to represent something.
    :return: The video.
    """

    fig = standard_matplotlib_figure()

    point_count = itertools.count()

    axis = fig.add_subplot(1, 1, 1)

    for current_values in zip(
        more_itertools.grouper(result_layers.result.data, frames_per_context),
        zip(
            *[
                more_itertools.grouper(layer.data, frames_per_context)
                for layer in result_layers.layers
            ]
        ),
    ):

        result_values = list(filter(None, current_values[0]))
        layers_values = list(list(filter(None, layer)) for layer in current_values[1])

        x_axis = np.arange(len(result_values))

        axis.plot(
            x_axis,
            result_values,
            color="red",
            label=result_layers.result.label,
            linewidth=5,
        )

        for label, values, color in zip(
            (layer.label for layer in result_layers.layers), layers_values, infinite_colors()
        ):
            axis.plot(
                x_axis, values, label=label, linewidth=3, linestyle="dashed", alpha=0.5, color=color
            )

        axis.set_title(title)
        axis.set_ylabel("Values")
        axis.set_xlabel("Point #")
        axis.grid()
        axis.legend(loc="upper right")

        all_values = list(
            filter(
                lambda value: not pd.isna(value),
                result_values + list(itertools.chain.from_iterable(layers_values)),
            )
        )
        y_min = min(all_values) - 10 if all_values else -10
        y_max = max(all_values) + 10 if all_values else 10

        axis.set_ylim(y_min, y_max)

        if horizontal_line is not None:
            axis.hlines(
                y=horizontal_line, xmin=0, xmax=max(x_axis), linestyles="dotted", colors="purple"
            )

        plt.tight_layout()

        for inter_group_index, _ in enumerate(result_values):

            line = axis.vlines(x=inter_group_index, ymin=y_min - 5, ymax=y_max + 5, color="red")

            LOGGER.info(
                f"Visualizing index #{next(point_count)} "
                f"of the layers of [{result_layers.result.label}]"
            )

            yield visualization_common.render_current_matplotlib_frame(
                fig=fig, resolution=(video_height, video_height)
            )

            line.remove()

        for axes in fig.axes:
            axes.clear()
