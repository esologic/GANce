"""
Test functions for some of the more complicated image pipelines.
"""

from test.assets import SAMPLE_BATCH_1_NETWORK_PATH, SAMPLE_BATCH_2_NETWORK_PATH, WAV_CLAPS_PATH
from typing import Tuple

import pytest

from gance.data_into_network_visualization import network_visualization
from gance.data_into_network_visualization.network_visualization import SynthesisOutput
from gance.data_into_network_visualization.visualization_inputs import (
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.gance_types import OptionalImageSourceType
from gance.image_sources import image_sources_common
from gance.network_interface.network_functions import MultiNetwork
from gance.vector_sources.music import read_wavs_scale_for_video
from gance.vector_sources.vector_sources_common import sub_vectors


def get_network_output(
    enable_3d: bool,
    enable_2d: bool,
    video_fps: float,
    network_enabled: bool,
    visualization_height: int,
) -> Tuple[int, SynthesisOutput]:
    """
    Modify only the test-relevant components of the visualization run.
    :param enable_3d: 3D network panel will be added.
    :param enable_2d: 2D network panel will be added.
    :param video_fps: FPS of resulting video stream.
    :param network_enabled: If network images will be added.
    :param visualization_height: Controls resolution of the output visualization.
    :return: Tuple, (number of expected output frames, streams of frames)
    """

    with MultiNetwork(
        network_paths=[SAMPLE_BATCH_1_NETWORK_PATH, SAMPLE_BATCH_2_NETWORK_PATH]
    ) as multi_networks:

        time_series_audio_vectors = read_wavs_scale_for_video(
            wavs=[WAV_CLAPS_PATH],
            vector_length=multi_networks.expected_vector_length,
            frames_per_second=video_fps,
        ).wav_data

        data = alpha_blend_vectors_max_rms_power_audio(
            alpha=0.5,  # Param doesn't matter for test.
            fft_roll_enabled=True,  # Param doesn't matter for test.
            fft_amplitude_range=(-10, 10),  # Param doesn't matter for test.
            time_series_audio_vectors=time_series_audio_vectors,
            vector_length=multi_networks.expected_vector_length,
            network_indices=multi_networks.network_indices,
        )

        return len(
            sub_vectors(data.combined.data, vector_length=multi_networks.expected_vector_length)
        ), network_visualization.vector_synthesis(
            data=data,
            networks=multi_networks if network_enabled else None,
            default_vector_length=multi_networks.expected_vector_length,
            enable_3d=enable_3d,
            enable_2d=enable_2d,
            visualization_height=visualization_height,
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
    "network_enabled",
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
    "visualization_height",
    [100, 300, 512, 1024],
)
def test_vector_synthesis_integration(
    enable_3d: bool,
    enable_2d: bool,
    video_fps: float,
    network_enabled: bool,
    visualization_height: int,
) -> None:
    """
    Integration style test, to verify that this function works as expected by examining
    the output.
    :param enable_3d: Passed to function under test.
    :param enable_2d: Passed to function under test.
    :param video_fps: Passed to function under test.
    :param network_enabled: Passed to function under test.
    :return: None
    """

    if not (enable_3d or enable_2d or network_enabled):
        with pytest.raises(ValueError):
            get_network_output(
                enable_3d=enable_3d,
                enable_2d=enable_2d,
                video_fps=video_fps,
                network_enabled=network_enabled,
                visualization_height=visualization_height,
            )
    else:
        expected_num_frames, network_output = get_network_output(
            enable_3d=enable_3d,
            enable_2d=enable_2d,
            video_fps=video_fps,
            network_enabled=network_enabled,
            visualization_height=visualization_height,
        )

        if enable_3d or enable_2d:
            visualization_frames = list(network_output.visualization_images)
            for frame in visualization_frames:
                resolution = image_sources_common.image_resolution(frame)
                assert resolution.height == visualization_height
                assert resolution.width == visualization_height * (enable_2d + enable_3d)
            assert expected_num_frames == len(visualization_frames)
        else:
            assert network_output.visualization_images is None

        if network_enabled:
            network_frames = list(network_output.synthesized_images)
            for frame in network_frames:
                resolution = image_sources_common.image_resolution(frame)
                assert resolution.height == 1024
                assert resolution.width == 1024
            assert expected_num_frames == len(network_frames)
        else:
            assert network_output.synthesized_images is None
