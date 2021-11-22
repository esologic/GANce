"""
CLI to do common things with a projection file.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import click

from gance.projection import projection_visualization
from gance.video_common import add_wav_to_video


@click.group()
def cli() -> None:
    """
    Use one of the commands below to process a projection file.

    \f

    :return: None
    """


@cli.command()
@click.option(
    "--projection_file",
    type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True),
    help="A Tuple (Path to the video to project, Path to destination of projection file)",
)
@click.option(
    "--video_path",
    type=click.Path(file_okay=True, writable=True, dir_okay=False, resolve_path=True),
    help="Destination path for the resulting visualization",
)
@click.option(
    "--audio_path",
    type=click.Path(file_okay=True, writable=True, dir_okay=False, resolve_path=True),
    default=None,
    help="If given, this audio file will be added to the resulting video.",
)
@click.option(
    "--video_height",
    type=click.IntRange(min=1),
    help="A Tuple (Path to the video to project, Path to destination of projection file)",
    default=1024,
    show_default=True,
)
def visualize_final_latents(  # pylint: disable=too-many-arguments,too-many-locals
    projection_file: str,
    video_path: str,
    audio_path: Optional[str],
    video_height: Optional[int],
) -> None:
    """
    Create a video that, side by side, displays:

    * Final Latents as a graph.
    * Target image.
    * Final latents image.

    \f

    :param projection_file: See click help or called function docs.
    :param video_path: See click help or called function docs.
    :param audio_path: Optional path to audio file.
    :param video_height: See click help or called function docs.
    :return: None
    """

    output_video_path = Path(video_path)

    with tempfile.NamedTemporaryFile(suffix=output_video_path.suffix) as f:

        tmp_video_path = Path(f.name)

        projection_visualization.visualize_final_latents(
            projection_file_path=Path(projection_file),
            output_video_path=tmp_video_path,
            video_height=video_height,
        )

        if audio_path is not None:

            while not tmp_video_path.exists():
                pass

            add_wav_to_video(
                video_path=tmp_video_path,
                audio_path=Path(audio_path),
                output_path=output_video_path,
            )
        else:
            shutil.move(src=str(tmp_video_path), dst=str(output_video_path))


@cli.command()
@click.option(
    "--projection_file",
    type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True),
    help="A Tuple (Path to the video to project, Path to destination of projection file)",
)
@click.option(
    "--video_path",
    type=click.Path(file_okay=True, writable=True, dir_okay=False, resolve_path=True),
    help="Destination path for the resulting visualization",
)
@click.option(
    "--audio_path",
    type=click.Path(file_okay=True, writable=True, dir_okay=False, resolve_path=True),
    default=None,
    help="If given, this audio file will be added to the resulting video.",
)
@click.option(
    "--video_height",
    type=click.IntRange(min=1),
    help="A Tuple (Path to the video to project, Path to destination of projection file)",
    default=1024,
    show_default=True,
)
def visualize_projection_of_frame(  # pylint: disable=too-many-arguments,too-many-locals
    projection_file: str,
    video_path: str,
    audio_path: Optional[str],
    video_height: Optional[int],
) -> None:
    """
    Create a video that, side by side, displays:

    * Final Latents as a graph.
    * Target image.
    * Final latents image.

    \f

    :param projection_file: See click help or called function docs.
    :param video_path: See click help or called function docs.
    :param audio_path: Optional path to audio file.
    :param video_height: See click help or called function docs.
    :return: None
    """

    output_video_path = Path(video_path)

    with tempfile.NamedTemporaryFile(suffix=output_video_path.suffix) as f:

        tmp_video_path = Path(f.name)

        projection_visualization.visualize_final_latents(
            projection_file_path=Path(projection_file),
            output_video_path=tmp_video_path,
            video_height=video_height,
        )

        if audio_path is not None:

            while not tmp_video_path.exists():
                pass

            add_wav_to_video(
                video_path=tmp_video_path,
                audio_path=Path(audio_path),
                output_path=output_video_path,
            )
        else:
            shutil.move(src=str(tmp_video_path), dst=str(output_video_path))


if __name__ == "__main__":
    cli()
