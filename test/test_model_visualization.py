"""
Test functions for some of the more complicated image pipelines.
"""

from test.assets import SAMPLE_BATCH_1_MODEL_PATH, SAMPLE_BATCH_2_MODEL_PATH, WAV_CLAPS_PATH
from typing import Tuple

import pytest

from gance.data_into_model_visualization import model_visualization
from gance.data_into_model_visualization.model_visualization import ModelOutput
from gance.data_into_model_visualization.visualization_inputs import (
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.gance_types import OptionalImageSourceType
from gance.image_sources import image_sources_common
from gance.model_interface.model_functions import MultiModel
from gance.vector_sources.music import read_wav_scale_for_video
from gance.vector_sources.vector_sources_common import sub_vectors


def get_model_output(
    enable_3d: bool, enable_2d: bool, video_fps: float, model_enabled: bool, video_side_length: int
) -> Tuple[int, ModelOutput]:
    """
    Modify only the test-relevant components of the visualization run.
    :param enable_3d: 3D model panel will be added.
    :param enable_2d: 2D model panel will be added.
    :param video_fps: FPS of resulting video stream.
    :param model_enabled: If model images will be added.
    :param video_side_length: Controls resolution of output video.
    :return: Tuple, (number of expected output frames, streams of frames)
    """

    with MultiModel(
        model_paths=[SAMPLE_BATCH_1_MODEL_PATH, SAMPLE_BATCH_2_MODEL_PATH]
    ) as multi_models:

        time_series_audio_vectors = read_wav_scale_for_video(
            wav=WAV_CLAPS_PATH,
            vector_length=multi_models.expected_vector_length,
            frames_per_second=video_fps,
        ).wav_data

        data = alpha_blend_vectors_max_rms_power_audio(
            alpha=0.5,  # Param doesn't matter for test.
            fft_roll_enabled=True,  # Param doesn't matter for test.
            fft_amplitude_range=(-10, 10),  # Param doesn't matter for test.
            time_series_audio_vectors=time_series_audio_vectors,
            vector_length=multi_models.expected_vector_length,
            model_indices=multi_models.model_indices,
        )

        return len(
            sub_vectors(data.combined.data, vector_length=multi_models.expected_vector_length)
        ), model_visualization.viz_model_ins_outs(
            data=data,
            models=multi_models if model_enabled else None,
            default_vector_length=multi_models.expected_vector_length,
            video_height=video_side_length,  # Param doesn't matter for test.
            enable_3d=enable_3d,
            enable_2d=enable_2d,
        )


def assert_all_none(frames: OptionalImageSourceType) -> int:
    """
    Assert that all frames in the input are None.
    :param frames: To check.
    :return: Number of frames in the iterator.
    """
    frame_list = [frame is None for frame in frames]
    assert all(frame_list)
    return len(frame_list)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_enabled",
    [True, False],
)
@pytest.mark.parametrize(
    "enable_3d",
    [True, False],
)
@pytest.mark.parametrize(
    "enable_2d",
    [True, False],
)
@pytest.mark.parametrize(
    "video_fps",
    [15.0, 30.0, 60.0],
)
@pytest.mark.parametrize(
    "video_side_length",
    [100, 300, 512, 1024],
)
def test_viz_model_ins_outs_integration(
    enable_3d: bool, enable_2d: bool, video_fps: float, model_enabled: bool, video_side_length: int
) -> None:
    """
    Integration style test, to verify that this function works as expected by examining
    the output.
    :param enable_3d: Passed to function under test.
    :param enable_2d: Passed to function under test.
    :param video_fps: Passed to function under test.
    :param model_enabled: Passed to function under test.
    :return: None
    """

    if not (enable_3d or enable_2d or model_enabled):
        with pytest.raises(ValueError):
            get_model_output(
                enable_3d=enable_3d,
                enable_2d=enable_2d,
                video_fps=video_fps,
                model_enabled=model_enabled,
                video_side_length=video_side_length,
            )
    else:
        expected_num_frames, model_output = get_model_output(
            enable_3d=enable_3d,
            enable_2d=enable_2d,
            video_fps=video_fps,
            model_enabled=model_enabled,
            video_side_length=video_side_length,
        )

        if enable_3d or enable_2d:
            visualization_frames = list(model_output.visualization_images)
            for frame in visualization_frames:
                resolution = image_sources_common.image_resolution(frame)
                assert resolution.height == video_side_length
                assert resolution.width == video_side_length * (enable_2d + enable_3d)
            assert expected_num_frames == len(visualization_frames)
        else:
            assert assert_all_none(model_output.visualization_images) == expected_num_frames

        if model_enabled:
            model_frames = list(model_output.model_images)
            for frame in model_frames:
                resolution = image_sources_common.image_resolution(frame)
                assert resolution.height == video_side_length
                assert resolution.width == video_side_length
            assert expected_num_frames == len(model_frames)
        else:
            assert assert_all_none(model_output.model_images) == expected_num_frames
