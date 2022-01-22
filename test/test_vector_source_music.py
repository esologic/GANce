from test.assets import WAV_CLAPS_PATH

import pytest

from gance.vector_sources import music


@pytest.mark.parametrize("multiplier", [2.0, 1.5, 0.3, 0.1, 10])
def test__scale_wav_to_sample_rate(multiplier: float) -> None:
    """

    :return:
    """

    original_wav = music.read_wav_file(WAV_CLAPS_PATH)
    scaled_wav = music._scale_wav_to_sample_rate(
        wav_file=original_wav, new_sample_rate=int(original_wav.sample_rate * multiplier)
    )
    assert int(len(original_wav.wav_data.data) * multiplier) == len(scaled_wav.wav_data.data)
