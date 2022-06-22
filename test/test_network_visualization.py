"""
Test functions for some of the more complicated image pipelines.
"""

import functools
from test.assets import SAMPLE_BATCH_1_NETWORK_PATH, SAMPLE_BATCH_2_NETWORK_PATH, WAV_CLAPS_PATH
from typing import List, Optional, Tuple, cast

import pytest

from gance.data_into_network_visualization import network_visualization
from gance.data_into_network_visualization.visualization_inputs import (
    VisualizationInput,
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.gance_types import OptionalImageSourceType, RGBInt8ImageType
from gance.image_sources import image_sources_common
from gance.network_interface.network_functions import MultiNetwork
from gance.vector_sources.music import read_wavs_scale_for_video
from gance.vector_sources.vector_sources_common import ConcatenatedVectors, sub_vectors


@functools.lru_cache(maxsize=100)
def load_vis_data(
    vector_length: int, fps: float, network_indices: Tuple[int, ...]
) -> VisualizationInput:
    """
    Load the input to the model.
    :param vector_length: Input from param test.
    :param fps: Input from param test.
    :param network_indices: Input from param test.
    :return: To eventually feed into model.
    """

    time_series_audio_vectors = cast(
        ConcatenatedVectors,
        read_wavs_scale_for_video(
            wavs=[WAV_CLAPS_PATH],
            vector_length=vector_length,
            frames_per_second=fps,
        ).wav_data,
    )

    return alpha_blend_vectors_max_rms_power_audio(
        alpha=0.5,  # Param doesn't matter for test.
        fft_roll_enabled=True,  # Param doesn't matter for test.
        fft_amplitude_range=(-10, 10),  # Param doesn't matter for test.
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        network_indices=list(network_indices),
    )


def get_network_output(
    enable_3d: bool,
    enable_2d: bool,
    video_fps: float,
    network_enabled: bool,
    visualization_height: int,
    force_optimize_synthesis_order: bool,
) -> Tuple[int, Tuple[List[RGBInt8ImageType], List[RGBInt8ImageType]]]:
    """
    Modify only the test-relevant components of the visualization run.
    :param enable_3d: 3D network panel will be added.
    :param enable_2d: 2D network panel will be added.
    :param video_fps: FPS of resulting video stream.
    :param network_enabled: If network images will be added.
    :param visualization_height: Controls resolution of the output visualization.
    :param force_optimize_synthesis_order: Controls synthesis order.
    :return: Tuple, (number of expected output frames, streams of frames)
    """

    with MultiNetwork(
        network_paths=[SAMPLE_BATCH_1_NETWORK_PATH, SAMPLE_BATCH_2_NETWORK_PATH]
    ) as multi_networks:

        data = load_vis_data(
            vector_length=multi_networks.expected_vector_length,
            fps=video_fps,
            network_indices=tuple(multi_networks.network_indices),
        )

        synthesis = network_visualization.vector_synthesis(
            data=data,
            networks=multi_networks if network_enabled else None,
            default_vector_length=multi_networks.expected_vector_length,
            enable_3d=enable_3d,
            enable_2d=enable_2d,
            visualization_height=visualization_height,
            force_optimize_synthesis_order=force_optimize_synthesis_order,
        )

        def convert_output(iterator: OptionalImageSourceType) -> Optional[List[RGBInt8ImageType]]:
            """
            Convert an iterator to a list if it is given, otherwise return None.
            :param iterator: To convert.
            :return: Iterator as a list, or None if no iterator is given.
            """
            return list(iterator) if iterator is not None else None

        # Convert these out to a list so we can consume them more than once.
        return len(
            sub_vectors(data.combined.data, vector_length=multi_networks.expected_vector_length)
        ), (
            convert_output(synthesis.visualization_images),
            convert_output(synthesis.synthesized_images),
        )


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
@pytest.mark.parametrize(
    "force_optimize_synthesis_order",
    [True, False],
)
def test_vector_synthesis_integration(
    enable_3d: bool,
    enable_2d: bool,
    video_fps: float,
    network_enabled: bool,
    visualization_height: int,
    force_optimize_synthesis_order: bool,
) -> None:
    """
    Integration style test, to verify that this function works as expected by examining
    the output.
    :param enable_3d: Passed to function under test.
    :param enable_2d: Passed to function under test.
    :param video_fps: Passed to function under test.
    :param network_enabled: Passed to function under test.
    :param force_optimize_synthesis_order: Passed into FUT.
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
                force_optimize_synthesis_order=force_optimize_synthesis_order,
            )
    else:
        expected_num_frames, (visualization_frames, network_frames) = get_network_output(
            enable_3d=enable_3d,
            enable_2d=enable_2d,
            video_fps=video_fps,
            network_enabled=network_enabled,
            visualization_height=visualization_height,
            force_optimize_synthesis_order=force_optimize_synthesis_order,
        )
        if enable_3d or enable_2d:
            for frame in visualization_frames:
                resolution = image_sources_common.image_resolution(frame)
                assert resolution.height == visualization_height
                assert resolution.width == visualization_height * (enable_2d + enable_3d)
            assert expected_num_frames == len(visualization_frames)
        else:
            assert visualization_frames is None

        if network_enabled:
            for frame in network_frames:
                resolution = image_sources_common.image_resolution(frame)
                assert resolution.height == 1024
                assert resolution.width == 1024
            assert expected_num_frames == len(network_frames)
        else:
            assert network_frames is None
