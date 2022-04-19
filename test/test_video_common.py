"""
Test of critical functions of video reader using known files.
"""

from pathlib import Path
from test.assets import (
    BATCH_2_IMAGE_1_PATH,
    SAMPLE_FACE_VIDEO_EXPECTED_FPS,
    SAMPLE_FACE_VIDEO_EXPECTED_FRAMES_COUNT,
    SAMPLE_FACE_VIDEO_PATH,
    SAMPLE_FACE_VIDEO_SHORT_PATH,
    WAV_CLAPS_PATH,
)
from typing import Optional

import numpy as np
import pytest
from py._path.local import LocalPath  # pylint: disable=protected-access

import gance.image_sources.still_image_common
from gance.image_sources import video_common


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

    image = np.array(gance.image_sources.still_image_common.read_image(BATCH_2_IMAGE_1_PATH))

    # Verified these magic numbers experimentally
    assert image.sum() == 299876727  # whole image
    assert image[0].sum() == 250099  # red channel


def test_add_wav_to_video(tmpdir: LocalPath) -> None:
    """
    Test to make sure these functions work.
    Manually verified that, at one point in time, these both worked as expected.
    :param tmpdir: Test fixture
    :return: None
    """

    temp_dir = Path(tmpdir)

    video_common.add_wav_to_video(
        video_path=SAMPLE_FACE_VIDEO_PATH,
        audio_path=WAV_CLAPS_PATH,
        output_path=temp_dir.joinpath("output_single.mp4"),
    )

    video_common.add_wavs_to_video(
        video_path=SAMPLE_FACE_VIDEO_PATH,
        audio_paths=[WAV_CLAPS_PATH, WAV_CLAPS_PATH, WAV_CLAPS_PATH],
        output_path=temp_dir.joinpath("output_double.mp4"),
    )


@pytest.mark.parametrize("test_video_file", [SAMPLE_FACE_VIDEO_PATH, SAMPLE_FACE_VIDEO_SHORT_PATH])
@pytest.mark.parametrize("high_quality", [True, False])
def test__create_video_writer_resolution(
    tmpdir: LocalPath, test_video_file: Path, high_quality: bool
) -> None:
    """
    Reads a video from disk, and then re-writes it using the standard output function to make sure
    thinks like resolution and framerate are maintained as expected.
    :param tmpdir: Test fixture.
    :param test_video_file: Path to the video to re-write.
    :param high_quality: Input flag.
    :return: None
    """
    video_frames = video_common.frames_in_video(video_path=test_video_file)

    output_path = Path(tmpdir).joinpath("output.mp4")

    writer = video_common._create_video_writer_resolution(  # pylint: disable=protected-access
        output_path,
        video_fps=video_frames.original_fps,
        resolution=video_frames.original_resolution,
        high_quality=high_quality,
    )

    original_frames = 0
    for frame in video_frames.frames:
        writer.write(frame)
        original_frames += 1

    writer.release()

    re_read = video_common.frames_in_video(video_path=output_path)

    assert len(list(re_read.frames)) == original_frames
    assert re_read.original_fps == video_frames.original_fps
    assert re_read.original_resolution == video_frames.original_resolution
