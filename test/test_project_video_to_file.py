"""
Tests the helper functions within the CLI
"""

from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from py._path.local import LocalPath  # pylint: disable=protected-access

from gance.cli_common import EXTENSION_HDF5
from project_video_to_file import (  # pylint: disable=protected-access
    _directory_of_io,
    _passed_directly,
    _VideoPathOutputPath,
)


@pytest.mark.parametrize(
    argnames="video_output,exception_expected",
    argvalues=[
        ([("./video", "./projection_file.hdf5")], False),
        ([("./video", "./projection_file.hdf5"), ("./video1", "./projection_file1.hdf5")], False),
        ([("./video", None)], True),
        ([(None, "./projection_file.hdf5")], True),
        ([], False),
        (None, True),
    ],
)
def test__passed_directly(
    video_output: Optional[List[Tuple[str, str]]], exception_expected: bool
) -> None:
    """
    Check to make sure the `video_output` param works correctly.
    :return: None
    """

    if exception_expected:
        with pytest.raises(Exception):
            _passed_directly(
                video_output=video_output,
            )
        # Don't care if the remaining parse works, just want to make sure this case causes a raise.
        return
    else:
        result = _passed_directly(
            video_output=video_output,
        )

    expected_result = [
        _VideoPathOutputPath(video_path=Path(video_path), output_path=Path(output_path))
        for video_path, output_path in video_output
    ]

    assert result == expected_result


@pytest.mark.parametrize(argnames="output_file_prefix", argvalues=["test_", "prod_projection"])
@pytest.mark.parametrize(argnames="video_extension", argvalues=["mp4", "avi"])
@pytest.mark.parametrize(
    argnames="good_videos,bad_videos", argvalues=[(5, 10), (5, 5), (10, 10), (1, 0), (0, 1)]
)
def test__io_pairs_directory(  # pylint: disable=too-many-locals
    tmpdir: LocalPath,
    video_extension: str,
    good_videos: int,
    bad_videos: int,
    output_file_prefix: str,
) -> None:
    """
    Check to make sure the directory passing works as expected.
    :return: None
    """

    tmp_dir = Path(tmpdir)

    videos_dir = tmp_dir.joinpath("videos")
    videos_dir.mkdir()

    def make_video(path: Path) -> Path:
        path.touch()
        return path

    good_video_paths = [
        make_video(videos_dir.joinpath(f"video_{i}.{video_extension}")) for i in range(good_videos)
    ]
    bad_video_paths = [
        make_video(videos_dir.joinpath(f"video_{i}.NOTAVIDEO")) for i in range(bad_videos)
    ]

    output_dir = tmp_dir.joinpath("output")
    output_dir.mkdir()

    output = _directory_of_io(
        directory_of_videos=str(videos_dir),
        video_extension=video_extension,
        output_file_directory=str(output_dir),
        output_file_prefix=output_file_prefix,
    )

    if good_video_paths:

        output_videos, output_file_paths = [
            list(unzipped_iterable) for unzipped_iterable in zip(*output)
        ]

        assert sorted(output_videos) == sorted(good_video_paths)
        assert not any(path in output_videos for path in bad_video_paths)

        assert len(set(output_file_paths)) == len(output_videos) == good_videos

        for video_file, output_file in output:
            assert EXTENSION_HDF5 in output_file.suffix
            assert output_file.parent == output_dir
            assert video_file.with_suffix("").name in output_file.name

    else:
        assert output == []
