"""
Test functions for some of the more complicated image pipelines.
"""

from pathlib import Path, PosixPath
from test.assets import SAMPLE_BATCH_1_MODEL_PATH, SAMPLE_BATCH_2_MODEL_PATH, WAV_CLAPS_PATH

from gance.assets import OUTPUT_DIRECTORY
from gance.data_into_model_visualization import model_visualization
from gance.data_into_model_visualization.visualization_inputs import (
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.image_sources import video_common
from gance.model_interface.model_functions import MultiModel
from gance.vector_sources.music import read_wav_scale_for_video


def test_viz_model_ins_outs_integration(
    tmp_path: PosixPath,  # pylint: disable=unused-argument
) -> None:
    """

    :param tmp_path:
    :return:
    """

    video_fps = 60.0
    vector_length = 512

    time_series_audio_vectors = read_wav_scale_for_video(
        WAV_CLAPS_PATH, vector_length, video_fps
    ).wav_data

    with MultiModel(
        model_paths=[SAMPLE_BATCH_1_MODEL_PATH, SAMPLE_BATCH_2_MODEL_PATH]
    ) as multi_models:

        model_output = model_visualization.viz_model_ins_outs(
            data=alpha_blend_vectors_max_rms_power_audio(
                alpha=0.5,
                fft_roll_enabled=True,
                fft_amplitude_range=(-10, 10),
                time_series_audio_vectors=time_series_audio_vectors,
                vector_length=vector_length,
                model_indices=multi_models.model_indices,
            ),
            models=multi_models,
            default_vector_length=vector_length,
            video_height=1024,
            enable_3d=False,
            enable_2d=False,
            frames_to_visualize=None,
        )

        video_common.write_source_to_disk(
            source=video_common.horizontal_concat_optional_sources([model_output.model_images]),
            video_path=Path(OUTPUT_DIRECTORY.joinpath("test_output.mp4")),
            video_fps=video_fps,
        )
