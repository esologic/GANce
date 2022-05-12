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


def main() -> None:

    queue = collections.deque(maxlen=50)

    with gance.network_interface.fast_synthesizer.fast_synthesizer(
        data_source=itertools.repeat(np.zeros(shape=(512,))),
        network_path=Path("gance/assets/networks/production_network.pkl"),
    ) as frames:

        for index, item in enumerate(frames):
            queue.append(datetime.datetime.now())

            if len(queue) == 50:
                print(len(queue) / ((queue[-1] - queue[0]).total_seconds()))


            if index == 200:
                raise ValueError("FUCKER!!")

        print("out here")

    print("further out here")


if __name__ == "__main__":
    main()
