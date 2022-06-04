"""
Temporary
"""

import collections
from pathlib import Path
from time import sleep

import numpy as np

import gance.network_interface.fast_synthesizer
from gance.image_sources import video_common
from gance.image_sources.video_common import ImageSourceType


def main() -> None:
    """
    Temporary entrypoint.
    :return: None
    """

    queue = collections.deque(maxlen=50)  # type: ignore  # pylint: disable=unused-variable

    def input_source() -> ImageSourceType:
        """
        Create an never-ending input source.
        :return: Unlimited randomized vectors.
        """

        while True:
            yield np.random.rand(512)

    with gance.network_interface.fast_synthesizer.fast_synthesizer(
        data_source=input_source(),
        network_path=Path("gance/assets/networks/production_network.pkl"),
    ) as frames:

        for _, _ in enumerate(video_common.display_frame_forward_opencv(frames, full_screen=True)):
            sleep(33 / 1000)


if __name__ == "__main__":
    main()
