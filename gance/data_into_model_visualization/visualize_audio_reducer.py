"""
Functions to visualize the vector reduction process.
"""

import itertools
from pathlib import Path

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from gance.data_into_model_visualization.visualization_common import (
    VectorsReducer,
    standard_matplotlib_figure,
)
from gance.dynamic_model_switching import model_index_selector
from gance.vector_sources.music import read_wav_scale_for_video


def visualize_reducer_output(audio_path: Path, reducer: VectorsReducer) -> None:
    """
    Demo to visualize RMS values of a known mp3.
    :return: None
    """

    vector_length = 1000

    audio = read_wav_scale_for_video(
        wav=audio_path, vector_length=vector_length, frames_per_second=60.0
    ).wav_data

    reduced = model_index_selector(
        time_series_audio_vectors=audio,
        model_indices=list(range(30)),
        reducer=reducer,
        vector_length=vector_length,
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
        itertools.cycle(
            # This is a deterministic way to get the same sequence of colors frame after
            # frame. There might be a cleaner way to do this.
            list(mcolors.BASE_COLORS.keys())
            + list(mcolors.TABLEAU_COLORS.keys())
        ),
    ):
        layers_axis.plot(x_values, layer.data, alpha=0.5, label=layer.label, color=str(color))

    layers_axis.legend(loc="upper right")
    layers_axis.set_title("Compositional Layers")
    layers_axis.set_ylabel("Signal Amplitude")
    layers_axis.set_xlabel("Frame #")

    # Plot the result
    result_axis.plot(x_values, reduced.result.data)
    result_axis.set_title("Quantized Result")
    result_axis.set_ylabel("Model Index")
    result_axis.set_xlabel("Frame #")

    fig.suptitle(f"Model Selection for {audio_path.name}", fontsize=16)

    plt.tight_layout()
    plt.show()
