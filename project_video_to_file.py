"""
CLI to project videos to a file.

# TODO: quite a bit of duplication here, but works so going to leave it for now.
"""

import logging
from pathlib import Path
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import click

import gance.projection.projector_file_writer
from gance.cli_common import DEFAULT_TWO_TUPLE, EXTENSION_HDF5, EXTENSION_MP4
from gance.logger_common import LOGGER, LOGGER_DATE_FORMAT, LOGGER_FORMAT


class _VideoPathOutputPath(NamedTuple):
    """
    Intermediate type, would be bad if these got swapped.
    """

    video_path: Path
    output_path: Path


def _passed_directly(video_output: Optional[List[Tuple[str, str]]]) -> List[_VideoPathOutputPath]:
    """
    Parse filenames passed directly into the canonical structure.
    :param video_output: List of tuples from click.
    :return: List of NT's
    """

    return [
        _VideoPathOutputPath(video_path=Path(video_path), output_path=Path(output_path))
        for video_path, output_path in video_output
    ]


def _directory_of_io(
    directory_of_videos: Optional[str],
    video_extension: Optional[str],
    output_file_directory: Optional[str],
    output_file_prefix: Optional[str],
) -> List[_VideoPathOutputPath]:
    """
    Search the input directory for videos to project, then create the output directory and
    paths for the output files.
    Args are forwarded directly from click.
    :param directory_of_videos: See click help.
    :param video_extension: See click help.
    :param output_file_directory: See click help.
    :param output_file_prefix: See click help.
    :raises: ValueError if any input args are missing.
    :return:
    """

    inputs = [directory_of_videos, video_extension, output_file_directory, output_file_prefix]

    output_expected = any(inputs)

    if not output_expected:
        return []

    if not all(inputs):
        raise ValueError("Missing arguments for directory mode!")

    discovered_videos: List[Path] = list(Path(directory_of_videos).glob(f"*.{video_extension}"))

    if discovered_videos:
        output_directory = Path(output_file_directory)
        output_directory.mkdir(exist_ok=True)

        return [
            _VideoPathOutputPath(
                video_path=video_path,
                output_path=output_directory.joinpath(
                    f"{output_file_prefix}{video_path.with_suffix('').name}.hdf5"
                ),
            )
            for video_path in discovered_videos
        ]
    else:
        return []


def _process_io(  # pylint: disable=too-many-arguments,too-many-locals
    io_pairs: List[_VideoPathOutputPath],
    video_fps: Optional[float],
    path_to_network: str,
    projection_width_height: Optional[Tuple[int, int]],
    projection_fps: Optional[float],
    steps_per_projection: Optional[int],
    num_frames_to_project: Optional[int],
    latents_histories_enabled: bool,
    noises_histories_enabled: bool,
    images_histories_enabled: bool,
    log: Optional[str],
) -> None:
    """
    Project the videos in `io_pairs`.
    :param io_pairs: Tuples of the input videos, and the path to their projection files.
    :param video_fps: Can be used to override the actual FPS of the input video.
    :param path_to_network: Path to the network to do the projection with.
    :param projection_width_height: Scale each frame of the video to this size before feeding it
    into projection.
    :param projection_fps: Down sample the video to be at this FPS. Note, can only be lower than the
    original FPS of the input video, and must evenly go into original FPS. If not given, projection
    will be done at the native FPS of the input video.
    :param steps_per_projection: The number of times the `.step()` function will be called for the
    projection in this run. Default is the value baked into the stylegan2 repo.
    :param num_frames_to_project: The number of frames to project. After the video has been
    resampled to the fps given by `projection_fps`, this many frames will be projected of that
    video.
    :param latents_histories_enabled: Records the intermediate latents seen during projection.
    :param noises_histories_enabled: If the noises used in each projection should be recorded.
    Warning! This will make the output file MASSIVE.
    :param images_histories_enabled: If the images over time throughout the projection should be
    recorded. Warning! This will make the output file MASSIVE.
    :param log: If given, logs generated during this run will be written to this path.
    :return: None
    """

    network_path = Path(path_to_network)

    if log is not None:
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOGGER_FORMAT, LOGGER_DATE_FORMAT))
        LOGGER.addHandler(file_handler)
        LOGGER.info("Start of run!")

    LOGGER.info(f"Projecting {len(io_pairs)} videos!")

    for input_path, output_path in io_pairs:
        LOGGER.info(f"\t{input_path} -> {output_path}")
        if output_path.exists:
            LOGGER.warning(f"Projection file already exists! Be careful! {output_path}")

    for index, io_pair in enumerate(io_pairs):
        gance.projection.projector_file_writer.project_video_to_file(
            path_to_video=io_pair.video_path,
            path_to_network=network_path,
            projection_file_path=io_pair.output_path,
            video_fps=video_fps,
            # Click needs the tuple of `None`'s we don't want that though.
            projection_width_height=None
            if projection_width_height == DEFAULT_TWO_TUPLE
            else projection_width_height,
            projection_fps=projection_fps,
            steps_per_projection=steps_per_projection,
            num_frames_to_project=num_frames_to_project,
            latents_histories_enabled=latents_histories_enabled,
            noises_histories_enabled=noises_histories_enabled,
            images_histories_enabled=images_histories_enabled,
            batch_number=index,
        )


def common_command_options(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Create a decorator that contains the common options seen in both commands.
    :param func: Function to decorate.
    :return: Decorated function.
    """

    for option in reversed(
        [
            click.option(
                "--video_fps",
                type=click.FloatRange(min=1),
                help="Used to override the FPS of the input video file.",
                default=None,
                show_default=True,
            ),
            click.option(
                "--path_to_network",
                type=click.Path(
                    exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True
                ),
                help="Path to the network to do the projection with.",
            ),
            click.option(
                "--projection_width_height",
                type=click.Tuple([click.IntRange(min=0), click.IntRange(min=0)]),
                help=(
                    "If given each frame of the input video will be scaled "
                    "to this size before being fed into projection."
                ),
                default=DEFAULT_TWO_TUPLE,
                show_default=True,
            ),
            click.option(
                "--projection_fps",
                type=click.FloatRange(min=0),
                help=(
                    "Down sample the video to be at this FPS. Note, can only be lower than the "
                    "original FPS of the input video, and must evenly go into original FPS. "
                    "If not given, projection will be done at the native FPS of the input video."
                ),
                default=None,
                show_default=True,
            ),
            click.option(
                "--steps_per_projection",
                type=click.IntRange(min=0),
                help=(
                    "The number of times the `.step()` function will be called for the "
                    "projection in this run. If not given, the baked in value in `stylegan2` will "
                    "be used."
                ),
                default=None,
                show_default=True,
            ),
            click.option(
                "--num_frames_to_project",
                type=click.IntRange(min=0),
                help=(
                    "The number of frames to project. After the video has been down sampled to the "
                    "fps given by `projection_fps`, this many frames will be projected of that "
                    "video. If not given, every frame in the down sampled input video will be "
                    "projected."
                ),
                default=None,
                show_default=True,
            ),
            click.option(
                "--latents_histories_enabled",
                type=click.BOOL,
                help="Records the intermediate latents seen during projection.",
                default=True,
                show_default=True,
            ),
            click.option(
                "--noises_histories_enabled",
                type=click.BOOL,
                help=(
                    "If the noises used in each projection should be recorded. "
                    "Warning! This will make the output file MASSIVE."
                ),
                default=False,
                show_default=True,
            ),
            click.option(
                "--images_histories_enabled",
                type=click.BOOL,
                help=(
                    "If the images over time throughout the projection should be recorded. "
                    "Warning! This will make the output file MASSIVE."
                ),
                default=False,
                show_default=True,
            ),
            click.option(
                "--log",
                type=click.Path(file_okay=True, writable=True, dir_okay=False, resolve_path=True),
                help=("If given, logs generated during this run will be written to this path."),
                default=None,
                show_default=True,
            ),
        ]
    ):
        func = option(func)

    return func


@click.group()
def cli() -> None:
    """
    Use one of the commands below to project either a list of individual videos, or a directory
    of videos.

    \f

    :return: None
    """


@cli.command()  # type: ignore
@click.option(
    "--video_output",
    type=click.Tuple(
        types=[
            click.Path(
                exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True
            ),
            click.Path(writable=True, resolve_path=True),
        ]
    ),
    help="A Tuple (Path to the video to project, Path to destination of projection file)",
    multiple=True,
)
@common_command_options
def videos(  # pylint: disable=too-many-arguments,too-many-locals
    video_output: Optional[List[Tuple[str, str]]],
    video_fps: Optional[float],
    path_to_network: str,
    projection_width_height: Optional[Tuple[int, int]],
    projection_fps: Optional[float],
    steps_per_projection: Optional[int],
    num_frames_to_project: Optional[int],
    latents_histories_enabled: bool,
    noises_histories_enabled: bool,
    images_histories_enabled: bool,
    log: Optional[str],
) -> None:
    """
    Project individual videos to given files.

    \f

    :param video_output: Tuples of (video, resulting projection file path).
    :param video_fps: Can be used to override the actual FPS of the input video.
    :param path_to_network: Path to the network to do the projection with.
    :param output_path: Path to output file.
    :param projection_width_height: Scale each frame of the video to this size before feeding it
    into projection.
    :param projection_fps: Down sample the video to be at this FPS. Note, can only be lower than the
    original FPS of the input video, and must evenly go into original FPS. If not given, projection
    will be done at the native FPS of the input video.
    :param steps_per_projection: The number of times the `.step()` function will be called for the
    projection in this run. Default is the value baked into the stylegan2 repo.
    :param num_frames_to_project: The number of frames to project. After the video has been
    resampled to the fps given by `projection_fps`, this many frames will be projected of that
    video.
    :param latents_histories_enabled: Records the intermediate latents seen during projection.
    :param noises_histories_enabled: If the noises used in each projection should be recorded.
    Warning! This will make the output file MASSIVE.
    :param images_histories_enabled: If the images over time throughout the projection should be
    recorded. Warning! This will make the output file MASSIVE.
    :param log: If given, logs generated during this run will be written to this path.
    :return: None
    """

    _process_io(
        io_pairs=_passed_directly(video_output),
        video_fps=video_fps,
        path_to_network=path_to_network,
        projection_width_height=projection_width_height,
        projection_fps=projection_fps,
        steps_per_projection=steps_per_projection,
        num_frames_to_project=num_frames_to_project,
        latents_histories_enabled=latents_histories_enabled,
        noises_histories_enabled=noises_histories_enabled,
        images_histories_enabled=images_histories_enabled,
        log=log,
    )


@cli.command()  # type: ignore
@click.option(
    "--directory_of_videos",
    type=click.Path(exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True),
    help="Directory that contains videos to project.",
)
@click.option(
    "--video_extension",
    type=str,
    help="The extension of the videos in the given directory.",
    default=EXTENSION_MP4,
    show_default=True,
)
@click.option(
    "--output_file_directory",
    type=click.Path(file_okay=False, writable=True, resolve_path=True),
    help="Resulting projection files will be written to this directory.",
)
@click.option(
    "--output_file_prefix",
    type=str,
    help=(
        "Resulting projection files will be named: "
        "{prefix}_{video filename without extension}." + EXTENSION_HDF5
    ),
    default="projection_of_",
    show_default=True,
)
@common_command_options
def directory(  # pylint: disable=too-many-arguments,too-many-locals
    directory_of_videos: str,
    video_extension: str,
    output_file_directory: str,
    output_file_prefix: str,
    video_fps: Optional[float],
    path_to_network: str,
    projection_width_height: Optional[Tuple[int, int]],
    projection_fps: Optional[float],
    steps_per_projection: Optional[int],
    num_frames_to_project: Optional[int],
    latents_histories_enabled: bool,
    noises_histories_enabled: bool,
    images_histories_enabled: bool,
    log: Optional[str],
) -> None:
    """
    Project all of the video files in the given directory.

    \f

    :param directory_of_videos: Directory that contains videos to project.
    :param video_extension: The extension of the videos in the given directory.
    :param output_file_directory: Resulting projection files will be written to this directory.
    :param output_file_prefix: Will be prepended to the output projection file names.
    :param video_fps: Can be used to override the actual FPS of the input video.
    :param path_to_network: Path to the network to do the projection with.
    :param output_path: Path to output file.
    :param projection_width_height: Scale each frame of the video to this size before feeding it
    into projection.
    :param projection_fps: Down sample the video to be at this FPS. Note, can only be lower than the
    original FPS of the input video, and must evenly go into original FPS. If not given, projection
    will be done at the native FPS of the input video.
    :param steps_per_projection: The number of times the `.step()` function will be called for the
    projection in this run. Default is the value baked into the stylegan2 repo.
    :param num_frames_to_project: The number of frames to project. After the video has been
    resampled to the fps given by `projection_fps`, this many frames will be projected of that
    video.
    :param latents_histories_enabled: Records the intermediate latents seen during projection.
    :param noises_histories_enabled: If the noises used in each projection should be recorded.
    Warning! This will make the output file MASSIVE.
    :param images_histories_enabled: If the images over time throughout the projection should be
    recorded. Warning! This will make the output file MASSIVE.
    :param log: If given, logs generated during this run will be written to this path.
    :return: None
    """

    _process_io(
        io_pairs=_directory_of_io(
            directory_of_videos=directory_of_videos,
            video_extension=video_extension,
            output_file_directory=output_file_directory,
            output_file_prefix=output_file_prefix,
        ),
        video_fps=video_fps,
        path_to_network=path_to_network,
        projection_width_height=projection_width_height,
        projection_fps=projection_fps,
        steps_per_projection=steps_per_projection,
        num_frames_to_project=num_frames_to_project,
        latents_histories_enabled=latents_histories_enabled,
        noises_histories_enabled=noises_histories_enabled,
        images_histories_enabled=images_histories_enabled,
        log=log,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
