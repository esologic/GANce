"""
CLI to do common things with a projection file.
"""

from pathlib import Path
from typing import Optional

import click

from gance.projection import projection_visualization


@click.group()
def cli() -> None:
    """
    Use one of the commands below to process a projection file.

    \f

    :return: None
    """


@cli.command()  # type: ignore
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
    "--video_height",
    type=click.IntRange(min=1),
    help="A Tuple (Path to the video to project, Path to destination of projection file)",
    default=1024,
    show_default=True,
)
def visualize_final_latents(  # pylint: disable=too-many-arguments,too-many-locals
    projection_file: str,
    video_path: Path,
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
    :param video_height: See click help or called function docs.
    :return: None
    """

    projection_visualization.visualize_final_latents(
        projection_file_path=Path(projection_file),
        output_video_path=Path(video_path),
        video_height=video_height,
    )


if __name__ == "__main__":
    cli()
