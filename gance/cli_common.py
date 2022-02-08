"""
Constants, functions seen across CLIs.
"""
import click

EXTENSION_HDF5 = "hdf5"
EXTENSION_MP4 = "mp4"
DEFAULT_TWO_TUPLE = (None, None)

CLICK_OUTPUT_PATH = click.Path(
    exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True
)
CLICK_INPUT_PATH = click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True)

single_projection_file_path = click.option(
    "--projection_file",
    type=CLICK_OUTPUT_PATH,
    help="The path to the projection file to read.",
)

video_path = click.option(
    "--video_path",
    type=CLICK_INPUT_PATH,
    help="Destination path for the resulting visualization",
)

audio_paths = click.option(
    "--audio_path",
    type=click.Path(file_okay=True, writable=True, dir_okay=False, resolve_path=True),
    default=None,
    help="If given, this audio file will be added to the resulting video. "
    "If multiple audio files are given they will be added to the video file "
    "in the order they're given, one after another.",
    multiple=True,
)

video_height = click.option(
    "--video_height",
    type=click.IntRange(min=1),
    help="A Tuple (Path to the video to project, Path to destination of projection file)",
    default=1024,
    show_default=True,
)
