"""
Temporary
"""

import collections
import datetime
import itertools
import multiprocessing
from pathlib import Path

import numpy as np

import gance.network_interface.fast_synthesizer
from gance.image_sources import video_common
from gance.image_sources.image_sources_common import ImageResolution
from gance.network_interface import network_functions


def main() -> None:
    """

    :return:
    """

    queue = collections.deque(maxlen=50)  # type: ignore

    def input_source():
        while True:
            yield np.random.rand(512)

    with gance.network_interface.fast_synthesizer.fast_synthesizer(
        data_source=input_source(),
        network_path=Path("gance/assets/networks/production_network.pkl"),
    ) as frames:

        for _, frame in enumerate(
            video_common.display_frame_forward(frames, display_resolution=ImageResolution(500, 500))
        ):
            queue.append(datetime.datetime.now())

            if len(queue) == 50:
                print(len(queue) / ((queue[-1] - queue[0]).total_seconds()))


if __name__ == "__main__":
    main()
