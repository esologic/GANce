"""
Temporary
"""

import collections
import datetime
import itertools
import multiprocessing
import typing
from pathlib import Path
from typing import Iterator

import numpy as np

import gance.data_into_network_visualization.visualize_data_source
import gance.network_interface.fast_synthesizer
from gance import apply_spectrogram, iterator_common
from gance.image_sources import video_common
from gance.image_sources.image_sources_common import ImageResolution
from gance.network_interface import network_functions
from gance.vector_sources.vector_sources_common import SingleVector


@typing.no_type_check
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


@typing.no_type_check
def main_multi() -> None:
    """

    :return:
    """

    p1 = multiprocessing.Process(target=create_many_images, args=(0,))
    p2 = multiprocessing.Process(target=create_many_images, args=(1,))

    p1.start()
    p2.start()


@typing.no_type_check
def main_gpu() -> None:
    """

    :return:
    """

    queue = collections.deque(maxlen=50)

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


@typing.no_type_check
def main() -> None:
    """

    :return:
    """

    def random_values() -> Iterator[SingleVector]:
        """

        :return:
        """

        x = np.linspace(-np.pi, np.pi, 512)

        while True:
            yield np.sin(x)

    top_input = random_values()

    (
        concatenated,
        concatenated_images,
    ) = gance.data_into_network_visualization.visualize_data_source.visualize_data_source(
        iterator_common.apply_to_chunk(func=np.concatenate, n=10, source=top_input),
        title_prefix="Concatenated",
        resolution=ImageResolution(width=500, height=500),
    )

    spectrogram = map(
        lambda data: apply_spectrogram.reshape_spectrogram_to_vectors(
            apply_spectrogram.compute_spectrogram(data=data, num_frequency_bins=512 * 10),
            vector_length=512,
        ),
        concatenated,
    )

    (
        _,
        spectrogram_images,
    ) = gance.data_into_network_visualization.visualize_data_source.visualize_data_source(
        iterator_common.items_per_second(spectrogram),
        title_prefix="Spectrogram",
        resolution=ImageResolution(width=500, height=500),
    )

    for index, _ in enumerate(
        zip(
            video_common.display_frame_forward_opencv(
                spectrogram_images, window_name="Spectrogram"
            ),
            video_common.display_frame_forward_opencv(
                concatenated_images, window_name="Concatenated Data"
            ),
        )
    ):
        if index == 200:
            break


if __name__ == "__main__":
    main()
