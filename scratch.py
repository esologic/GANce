"""
Temporary
"""

import collections
import datetime
import itertools
import multiprocessing
from pathlib import Path

import more_itertools
import numpy as np

import gance.network_interface.fast_synthesizer
from gance import iterator_common
from gance.data_into_network_visualization import vectors_to_image
from gance.image_sources import video_common
from gance.image_sources.image_sources_common import ImageResolution
from gance.network_interface import network_functions


def create_many_images(gpu: int) -> None:
    """

    :param gpu:
    :return:
    """

    interface = network_functions.create_network_interface(
        network_path=Path("gance/assets/networks/production_network.pkl"),
        call_init_function=True,
        gpu_index=gpu,
    )

    vector = np.zeros(shape=(512,))

    while True:
        interface.create_image_vector(data=vector)


def main_multi() -> None:
    """

    :return:
    """

    p1 = multiprocessing.Process(target=create_many_images, args=(0,))
    p2 = multiprocessing.Process(target=create_many_images, args=(1,))

    p1.start()
    p2.start()


def main_gpu() -> None:
    """

    :return:
    """

    queue = collections.deque(maxlen=50)  # type: ignore

    with gance.network_interface.fast_synthesizer.fast_synthesizer(
        data_source=itertools.repeat(np.zeros(shape=(512,))),
        network_path=Path("gance/assets/networks/production_network.pkl"),
        num_gpus=4,
    ) as frames:

        for _, _ in enumerate(frames):
            queue.append(datetime.datetime.now())

            if len(queue) == 50:
                print(len(queue) / ((queue[-1] - queue[0]).total_seconds()))

        print("out here")

    print("further out here")


def main() -> None:
    """

    :return:
    """

    iterator = itertools.repeat(np.zeros(shape=(512,)))
    timed = iterator_common.items_per_second(iterator)

    data, images = vectors_to_image.visualize_data_source(
        timed, resolution=ImageResolution(width=500, height=500)
    )
    images = video_common.display_frame_forward(images)

    more_itertools.consume(images)


if __name__ == "__main__":
    main()
