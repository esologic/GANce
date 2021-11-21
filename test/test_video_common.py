"""
Test of critical functions of video reader using known files.
"""

from test.assets import (
    BATCH_2_IMAGE_1_PATH,
    SAMPLE_FACE_VIDEO_EXPECTED_FPS,
    SAMPLE_FACE_VIDEO_EXPECTED_FRAMES_COUNT,
    SAMPLE_FACE_VIDEO_PATH,
)
from typing import Optional

import numpy as np
import pytest

from gance import video_common


@pytest.mark.parametrize(
    "reduce_fps_to,expected_num_frames,expected_error",
    [
        (SAMPLE_FACE_VIDEO_EXPECTED_FPS, SAMPLE_FACE_VIDEO_EXPECTED_FRAMES_COUNT, None),
        (SAMPLE_FACE_VIDEO_EXPECTED_FPS / 2, SAMPLE_FACE_VIDEO_EXPECTED_FRAMES_COUNT / 2, None),
        (SAMPLE_FACE_VIDEO_EXPECTED_FPS / 4, SAMPLE_FACE_VIDEO_EXPECTED_FRAMES_COUNT / 4, None),
        (SAMPLE_FACE_VIDEO_EXPECTED_FPS / 2 + 0.1, None, ValueError),
    ],
)
def test_frames_in_video(
    reduce_fps_to: float, expected_num_frames: Optional[int], expected_error: Optional[Exception]
) -> None:
    """
    Test video reader creation.
    :param reduce_fps_to: Output frames will be read at this rate.
    :param expected_num_frames: Given `reduce_fps_to` and the known fps of the test asset we
    expect this number of frames.
    :param expected_error: If an exception is expected place it here.
    :return: None
    """

    if expected_error is None:

        video_frames = video_common.frames_in_video(
            video_path=SAMPLE_FACE_VIDEO_PATH, reduce_fps_to=reduce_fps_to
        )

        assert expected_num_frames == len(list(video_frames.frames))

    else:
        with pytest.raises(expected_error):  # type:ignore[call-overload]
            video_common.frames_in_video(
                video_path=SAMPLE_FACE_VIDEO_PATH, reduce_fps_to=reduce_fps_to
            )


def test_read_image() -> None:
    """
    Simple check to make sure function is working using a known image.
    :return: None
    """

    image = np.array(video_common.read_image(BATCH_2_IMAGE_1_PATH))

    # Verified these magic numbers experimentally
    assert image.sum() == 299876727  # whole image
    assert image[0].sum() == 250099  # red channel
