"""
Given some styleGAN networks, load each network and synthesize a number of images with them.
Interesting vectors can be reused with other networks.
"""
from pathlib import Path
from time import sleep

import click
import numpy as np

from gance.assets import PRODUCTION_NETWORK_PATH
from gance.gance_types import ImageSourceType
from gance.image_sources import video_common
from gance.network_interface import fast_synthesizer


@click.group()
def cli() -> None:
    """
    Tools to synthesize images from styleGAN networks in real-time.

    \f

    :return: None
    """


@cli.command()
@click.option(
    "--network",
    type=click.Path(file_okay=True, dir_okay=False, readable=True, resolve_path=True),
    default=str(PRODUCTION_NETWORK_PATH),
    help="Network pickle to use to synthesize images.",
    show_default=True,
)
@click.option(
    "--num-gpus",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Controls how many GPUs are used to synthesize images. "
        "By default, the max num possible GPUs will be used."
    ),
    show_default=True,
)
@click.option(
    "--fullscreen",
    is_flage=True,
    default=False,
    help="If given, the images will be displayed in a fullscreen window.",
    show_default=True,
)
def random(network: str, num_gpus: int, fullscreen: bool) -> None:
    """
    Synthesize random images with the input network as fast as possible. Resulting images are drawn
    onto a window on the host.

    \f

    :param network: See click docs for help.
    :param num_gpus: See click docs for help.
    :param fullscreen: See click docs for help.
    :return: None
    """

    def input_source() -> ImageSourceType:
        """
        Create an never-ending input source.
        :return: Unlimited randomized vectors.
        """

        while True:
            yield np.random.rand(512)

    with fast_synthesizer.fast_synthesizer(
        data_source=input_source(),
        network_path=Path(network),
        num_gpus=num_gpus,
    ) as frames:
        for _, _ in enumerate(
            video_common.display_frame_forward_opencv(frames, full_screen=fullscreen)
        ):
            sleep(33 / 1000)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
