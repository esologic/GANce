"""
Tests to make sure rms reduction works as expected.
"""

from test.assets import WAV_CLAPS_PATH

import numpy as np

from gance.vector_sources.music import read_wavs_scale_for_video
from gance.vector_sources.vector_reduction import reduce_vector_rms_rolling_max
from gance.vector_sources.vector_sources_common import sub_vectors
from gance.vector_sources.vector_types import ConcatenatedVectors


def test_reduce_vector_rms_alignment() -> None:
    """
    Checks to make sure that the value, and returned shape of the rms reduction function is as
    expected given a known piece of audio.
    :return: None
    """

    vector_length = 1000

    audio = ConcatenatedVectors(
        read_wavs_scale_for_video(
            wavs=[WAV_CLAPS_PATH], vector_length=vector_length, frames_per_second=60.0
        ).wav_data
    )

    single_audio_vector = sub_vectors(data=audio, vector_length=vector_length)[0]

    reduced = reduce_vector_rms_rolling_max(
        time_series_audio_vectors=single_audio_vector, vector_length=vector_length
    )

    assert reduced.result.data.shape[0] == 1

    # Known value.
    assert np.isclose(0.00298562, reduced.result.data[0])
