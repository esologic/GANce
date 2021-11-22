"""
Tests for synthesis file module.
"""

from hashlib import md5
from test.assets import SAMPLE_SYNTHESIS_FILE_PATH

from gance import synthesis_file


def test_read_vector_in_file_content() -> None:
    """
    Read a known file and hash it, then compare it to a known value.
    :return: None
    """
    assert (
        md5(synthesis_file.read_vector_in_file(SAMPLE_SYNTHESIS_FILE_PATH).tobytes()).hexdigest()
        == "ec0b12c590fc748668aadd260664284a"  # Hand verified that this is correct.
    )


def test_read_vector_in_file_shape() -> None:
    """
    Make sure the read data is shaped correctly, which isn't checked by hash.
    :return: None
    """
    assert synthesis_file.read_vector_in_file(SAMPLE_SYNTHESIS_FILE_PATH).shape == (512,)
