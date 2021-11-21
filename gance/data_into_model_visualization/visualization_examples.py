"""
Known, working usages some of the visualization functions.
"""
from time import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile

from gance import assets
from gance.apply_spectrogram import (
    compute_spectrogram,
    compute_spectrogram_smooth_scale,
    reshape_spectrogram_to_vectors,
)
from gance.assets import WAV_CLAPS_PATH
from gance.data_into_model_visualization.model_visualization import (
    _configure_axes,
    _frame_inputs,
    _write_data_to_axes,
)
from gance.data_into_model_visualization.vectors_to_image import (
    multi_plot_vectors,
    vectors_to_video,
    visualize_data_with_spectrogram_and_3d_vectors,
)
from gance.data_into_model_visualization.visualization_common import (
    STANDARD_MATPLOTLIB_DPI,
    STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE,
)
from gance.data_into_model_visualization.visualization_inputs import (
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.data_into_model_visualization.visualize_audio_reducer import visualize_reducer_output
from gance.dynamic_model_switching import reduce_vector_gzip_compression_rolling_average
from gance.vector_sources.music import read_wav_scale_for_video
from gance.vector_sources.vector_sources_common import (
    rotate_vectors_over_time,
    smooth_across_vectors,
)
from gance.vector_sources.vector_types import ConcatenatedVectors, VectorsLabel


def demo_smoothing() -> None:
    """
    Plots two spectrograms, one with smoothed vectors and one with un-smoothed vectors.
    :return: None
    """

    vector_length = 1000
    _, audio = wavfile.read(WAV_CLAPS_PATH)
    spectrogram = reshape_spectrogram_to_vectors(
        spectrogram_data=compute_spectrogram(ConcatenatedVectors(audio), vector_length),
        vector_length=vector_length,
    )
    smooth_spectrogram = smooth_across_vectors(spectrogram, vector_length)
    multi_plot_vectors(
        [
            VectorsLabel(spectrogram, vector_length, "Spectrogram"),
            VectorsLabel(smooth_spectrogram, vector_length, "Smooth Spectrogram"),
        ],
    )


def demo_visualize_data_with_spectrogram_and_3d_vectors() -> None:
    """
    Plot a few transformations for debugging.
    :return: None
    """
    _, audio = wavfile.read(WAV_CLAPS_PATH)
    visualize_data_with_spectrogram_and_3d_vectors(
        vectors_label=VectorsLabel(data=audio, vector_length=1000, label="Claps")
    )


def demo_visualize_reducer_output() -> None:
    """
    Visualize how a song is reduced, you could switch up the song or the reduction function.
    :return: None
    """
    visualize_reducer_output(WAV_CLAPS_PATH, reduce_vector_gzip_compression_rolling_average)


def data_visualizations_single_frame() -> None:
    """
    Demo function, draws a single frame of a data visualization pane.
    Mostly for test/debug.
    :return: None
    """

    vector_length = 1000

    # Needs to be this aspect ratio
    fig = plt.figure(
        figsize=(STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE, STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE),
        dpi=STANDARD_MATPLOTLIB_DPI,
        constrained_layout=False,
    )

    data = alpha_blend_vectors_max_rms_power_audio(
        time_series_audio_vectors=read_wav_scale_for_video(
            WAV_CLAPS_PATH, vector_length, 60.0
        ).wav_data,
        vector_length=vector_length,
        model_indices=list(np.arange(20)),
    )

    configured_axes = _configure_axes(
        fig=fig,
        enable_2d=True,
        enable_3d=False,
        visualization_input=data,
        vector_length=vector_length,
    )

    inputs = _frame_inputs(visualization_input=data, vector_length=vector_length)

    drawn_elements = _write_data_to_axes(
        axes=configured_axes,
        frame_input=inputs[18],
        vector_length=vector_length,
    )

    plt.show()

    for element in drawn_elements:
        element.remove()


def demo_rotation() -> None:
    """

    :return:
    """

    vector_length = 512

    time_series_audio_vectors = read_wav_scale_for_video(
        WAV_CLAPS_PATH, vector_length, 60.0
    ).wav_data

    spectrogram = compute_spectrogram_smooth_scale(
        data=time_series_audio_vectors,
        vector_length=vector_length,
        amplitude_range=(-10, 10),
    )

    rotated = rotate_vectors_over_time(
        data=spectrogram, vector_length=vector_length, roll_values=np.full((1000,), 10)
    )

    labeled_rotated = VectorsLabel(data=rotated, vector_length=512, label="Rotated")

    multi_plot_vectors(
        vectors_labels=[
            VectorsLabel(data=spectrogram, vector_length=512, label="Raw Spectrogram"),
            labeled_rotated,
        ]
    )

    output_path = assets.OUTPUT_DIRECTORY.joinpath(f"rotation_example_{int(time())}.mp4")

    print(output_path)

    vectors_to_video(
        labeled_data=labeled_rotated, output_path=output_path, video_height=1000, video_fps=15.0
    )


if __name__ == "__main__":
    demo_rotation()
