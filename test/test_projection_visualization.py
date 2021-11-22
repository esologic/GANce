"""
Runs the projection file visualization functions.
If these are able to complete it's a good indicator that the underlying functionality is
still working. It might be producing bad results but at least it's running.
"""
from pathlib import Path
from test.assets import SAMPLE_PROJECTION_FILE_PATH
from test.test_common import wait_get_size

from py._path.local import LocalPath  # pylint: disable=protected-access

from gance.projection import projection_visualization


def test_visualize_projection_convergence(tmpdir: LocalPath) -> None:
    """
    This should run without error.
    :param tmpdir: Test fixture.
    :return: None
    """
    file = Path(str(tmpdir)).joinpath("output.png")
    projection_visualization.visualize_projection_convergence(
        projection_file_path=SAMPLE_PROJECTION_FILE_PATH,
        output_image_path=file,
    )
    # Verified that this looked good visually.
    assert wait_get_size(file) >= 240000


def test_visualize_final_latents(tmpdir: LocalPath) -> None:
    """
    This should run without error.
    :param tmpdir: Test fixture.
    :return: None
    """
    file = Path(str(tmpdir)).joinpath("output.mp4")
    projection_visualization.visualize_final_latents(
        projection_file_path=SAMPLE_PROJECTION_FILE_PATH,
        output_video_path=file,
    )
    # Verified that this looked good visually.
    assert wait_get_size(file) >= 600000
