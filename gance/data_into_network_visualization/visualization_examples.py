"""
Known, working usages some of the visualization functions.
"""
import tempfile
from pathlib import Path
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
from gance.data_into_network_visualization import network_visualization, visualization_common
from gance.data_into_network_visualization.network_visualization import (
    _configure_axes,
    _frame_inputs,
    _write_data_to_axes,
)
from gance.data_into_network_visualization.vectors_to_image import (
    multi_plot_vectors,
    vectors_to_video,
    visualize_data_with_spectrogram_and_3d_vectors,
)
from gance.data_into_network_visualization.visualization_inputs import (
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.data_into_network_visualization.visualize_vector_reduction import (
    visualize_reducer_output,
)
from gance.image_sources.video_common import add_wav_to_video
from gance.network_interface.network_functions import create_network_interface
from gance.projection import projection_file_reader
from gance.projection.projection_visualization import visualize_projection_history
from gance.vector_sources import music, primatives, vector_sources_common
from gance.vector_sources.music import read_wavs_scale_for_video
from gance.vector_sources.vector_reduction import reduce_vector_gzip_compression_rolling_average
from gance.vector_sources.vector_sources_common import (
    rotate_vectors_over_time,
    smooth_across_vectors,
)
from gance.vector_sources.vector_types import ConcatenatedVectors, SingleVector, VectorsLabel


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
    fig = visualization_common.standard_matplotlib_figure()

    data = alpha_blend_vectors_max_rms_power_audio(
        time_series_audio_vectors=read_wavs_scale_for_video(
            wavs=[WAV_CLAPS_PATH], vector_length=vector_length, frames_per_second=60.0
        ).wav_data,
        vector_length=vector_length,
        network_indices=list(np.arange(20)),
        alpha=0.5,
        fft_roll_enabled=False,
        fft_amplitude_range=(-4, 4),
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
    Shows the effects of rotating a concatenated.
    :return: None
    """

    vector_length = 512

    time_series_audio_vectors = read_wavs_scale_for_video(
        wavs=[WAV_CLAPS_PATH], vector_length=vector_length, frames_per_second=60.0
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


def blog_post_media() -> None:
    """
    Shows the creation of example images on the blog: https://www.esologic.com/gance/
    :return: None
    """

    output_dir = assets.OUTPUT_DIRECTORY.joinpath("final_blog_post_images")
    output_dir.mkdir(exist_ok=True)

    y_range = (-20, 20)

    network_interface = create_network_interface(
        network_path=assets.PRODUCTION_NETWORK_PATH, call_init_function=True
    )

    projection_file_latents = projection_file_reader.final_latents_at_frame(
        projection_file_path=assets.PROJECTION_FILE_PATH,
        frame_number=561,
    )

    network_visualization.single_vector_single_network_visualization(
        vector=SingleVector(projection_file_latents),
        title="Projection File Original Final Latents",
        output_image_path=output_dir.joinpath("projection_final_original.png"),
        network=network_interface,
        y_range=(-20, 20),
    )

    network_visualization.single_vector_single_network_visualization(
        vector=SingleVector(projection_file_latents * 0.9),
        title="Projection File Original Final Latents",
        output_image_path=output_dir.joinpath("projection_final_small.png"),
        network=network_interface,
        y_range=(-20, 20),
    )

    network_visualization.single_vector_single_network_visualization(
        vector=SingleVector(projection_file_latents * 1.1),
        title="Projection File Original Final Latents",
        output_image_path=output_dir.joinpath("projection_final_large.png"),
        network=network_interface,
        y_range=(-20, 20),
    )

    network_visualization.single_vector_single_network_visualization(
        vector=SingleVector(
            np.full(shape=(network_interface.expected_vector_length,), fill_value=10)
        ),
        title="Line",
        output_image_path=output_dir.joinpath("line_to_image.png"),
        network=network_interface,
        y_range=y_range,
    )

    sin_vector = np.sin(np.arange(0, network_interface.expected_vector_length / 10, 0.1)) * 10

    network_visualization.single_vector_single_network_visualization(
        vector=sin_vector,
        title="Sine Wave",
        output_image_path=output_dir.joinpath("sine_wav_to_image.png"),
        network=network_interface,
        y_range=y_range,
    )

    random_noise = np.random.rand(network_interface.expected_vector_length) * 10

    network_visualization.single_vector_single_network_visualization(
        vector=random_noise,
        title="Noise",
        output_image_path=output_dir.joinpath("noise_image.png"),
        network=network_interface,
        y_range=y_range,
    )

    network_visualization.single_vector_single_network_visualization(
        vector=primatives.single_square_wave_vector(
            rising_edge_x=150,
            falling_edge_x=500,
            y_offset=0,
            y_amplitude=10,
            vector_length=network_interface.expected_vector_length,
        ),
        title="Square Wave",
        output_image_path=output_dir.joinpath("original_step.png"),
        network=network_interface,
        y_range=y_range,
    )

    square_vector = primatives.single_square_wave_vector(
        rising_edge_x=160,
        falling_edge_x=500,
        y_offset=0,
        y_amplitude=10,
        vector_length=network_interface.expected_vector_length,
    )

    network_visualization.single_vector_single_network_visualization(
        vector=square_vector,
        title="Tweaked Square",
        output_image_path=output_dir.joinpath("modified_step.png"),
        network=network_interface,
        y_range=(-20, 20),
    )

    network_visualization.single_vector_single_network_visualization(
        vector=random_noise,
        title="Early training network",
        output_image_path=output_dir.joinpath("early_network_noise_image.png"),
        network=create_network_interface(
            network_path=assets.EARLY_TRAINING_NETWORK_PATH, call_init_function=True
        ),
        y_range=y_range,
    )

    network_visualization.single_vector_single_network_visualization(
        vector=random_noise,
        title="Middle training network",
        output_image_path=output_dir.joinpath("middle_network_noise_image.png"),
        network=create_network_interface(
            network_path=assets.MID_TRAINING_NETWORK_PATH, call_init_function=True
        ),
        y_range=y_range,
    )

    network_visualization.vectors_single_network_visualization(
        vectors_label=VectorsLabel(
            data=vector_sources_common.interpolate_between_vectors(
                start=sin_vector, end=square_vector, count=1000
            ),
            vector_length=network_interface.expected_vector_length,
            label="Interpolation between Sine vector and Square Wave",
        ),
        output_video_path=output_dir.joinpath("sin_square_interp.mp4"),
        network=network_interface,
        y_range=(-20, 20),
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:

        tmp_video_path = Path(f.name)

        network_visualization.vectors_single_network_visualization(
            vectors_label=VectorsLabel(
                data=music.read_wavs_scale_for_video(
                    wavs=[assets.NOVA_SNIPPET_PATH],
                    vector_length=network_interface.expected_vector_length,
                    frames_per_second=60.0,
                ).wav_data,
                vector_length=network_interface.expected_vector_length,
                label="Audio Directly Into Network",
            ),
            output_video_path=tmp_video_path,
            network=network_interface,
        )

        while not tmp_video_path.exists():
            pass

        add_wav_to_video(
            video_path=tmp_video_path,
            audio_path=Path(assets.NOVA_SNIPPET_PATH),
            output_path=output_dir.joinpath("direct_audio.mp4"),
        )

    visualize_projection_history(
        projection_file_path=assets.PROJECTION_FILE_PATH,
        output_video_path=output_dir.joinpath("projection_history.mp4"),
        projection_network_path=assets.PRODUCTION_NETWORK_PATH,
        network_not_matching_ok=False,
        start_frame_index=561,
        end_frame_index=562,
    )


if __name__ == "__main__":
    data_visualizations_single_frame()
