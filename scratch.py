"""
Temporary
"""

import multiprocessing
from pathlib import Path

import numpy as np

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


def main() -> None:
    """

    :return:
    """

    p1 = multiprocessing.Process(target=create_many_images, args=(0,))
    p2 = multiprocessing.Process(target=create_many_images, args=(1,))

    p1.start()
    p2.start()


if __name__ == "__main__":
    main()
