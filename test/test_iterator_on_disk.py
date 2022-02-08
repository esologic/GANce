"""
Test to make sure the cache functionality works as expected.
"""

from test.assets import SAMPLE_FACE_VIDEO_PATH
from typing import Any, List

import numpy as np
import pytest

from gance.image_sources import video_common
from gance.iterator_on_disk import iterator_on_disk


@pytest.mark.parametrize("copies", range(1, 4))
@pytest.mark.parametrize(
    "to_duplicate",
    [
        ["a", "screaming", "across", "the", "sky"],
        [0, 1, 2, 3],
        list(
            video_common.frames_in_video(
                video_path=SAMPLE_FACE_VIDEO_PATH, width_height=(10, 10)
            ).frames
        ),
    ],
)
def test_iterator_on_disk(to_duplicate: List[Any], copies: int) -> None:
    """
    Test with a few different inputs, of type and length, make sure the resulting iterators are
    all the same.
    :param to_duplicate: Passed to function, this is the iterator to cache.
    :param copies: Passed to function, this is the number of copies to produce.
    :return: None
    """

    result = iterator_on_disk(iterator=iter(to_duplicate), copies=copies)
    primary = result[0]
    secondaries = result[1:]
    assert len(secondaries) == copies
    assert np.array_equal(to_duplicate, list(primary))
    for secondary in secondaries:
        values = list(secondary)
        assert np.array_equal(to_duplicate, values)
