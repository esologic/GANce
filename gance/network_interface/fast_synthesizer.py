"""
Functionality to spread the synthesis of some input across multiple GPUs to get the highest
throughput.

Some very _very_ weird code here, using the global. This is the canonical way to pre-load an
expensive multiprocessing operation though if you can believe it or not.
"""
import logging
import multiprocessing
from contextlib import contextmanager
from functools import partial
from itertools import count
from multiprocessing import Queue, Semaphore
from pathlib import Path
from typing import Iterator, Optional, Union, overload

import nvsmi
from pympler import muppy, summary

from gance import iterator_common
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.network_interface.network_functions import NetworkInterface, create_network_interface
from gance.vector_sources.vector_sources_common import SingleMatrix, SingleVector, is_vector

# Creates a new one of these in each `multiprocessing.Pool` child thread. That way we can store
# large objects that will also be accessible in the `i/map(func...` call.
_network_interface: Optional[NetworkInterface] = None


_output_control_semaphore: Optional[Semaphore] = None

_frame_count: count = count()


@overload
def synthesize_frame(data_is_vector: bool, data: SingleVector) -> RGBInt8ImageType:
    ...


@overload
def synthesize_frame(data_is_vector: bool, data: SingleMatrix) -> RGBInt8ImageType:
    ...


def synthesize_frame(
    data_is_vector: bool, data: Union[SingleVector, SingleMatrix]
) -> RGBInt8ImageType:
    """
    Function responsible for converting data into images.
    This will be called within the child process in the `Pool`, so the `global` namespace is that
    of inside the pool, not in the main thread.
    :param data_is_vector: Decides which image function to use.
    :param data: To convert.
    :return: The resulting image.
    """

    _output_control_semaphore.acquire()

    frame_count = next(_frame_count)

    if frame_count == 200:
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)

    output = (
        _network_interface.create_image_vector(data)
        if data_is_vector
        else _network_interface.create_image_matrix(data)
    )

    return output


def initializer(
    network_path: Path, id_queue: "Queue[int]", output_control_semaphore: Semaphore
) -> None:
    """
    Called before data can be fed into the map function, let's us load the network as a global.
    This will be called within the child process in the `Pool`, so the `global` namespace is that
    of inside the pool, not in the main thread.
    :param network_path: Path to the network to load.
    :param id_queue: Contains unused GPU indices, one is selected for this process.
    :param output_control_semaphore:
    :return: None
    """

    gpu_index = id_queue.get()
    current = multiprocessing.current_process()
    logging.info(f"Creating network interface in: {current.pid}, using GPU index: {gpu_index}")

    global _network_interface  # pylint: disable=global-statement
    _network_interface = create_network_interface(
        network_path=network_path,
        gpu_index=gpu_index,
        call_init_function=True,
    )

    global _output_control_semaphore  # pylint: disable=global-statement
    _output_control_semaphore = output_control_semaphore


@contextmanager
def fast_synthesizer(
    data_source: Union[Iterator[SingleVector], Iterator[SingleMatrix]],
    network_path: Path,
    num_gpus: Optional[int] = None,
) -> Iterator[ImageSourceType]:
    """
    Split the synthesis of the input data source across multiple GPUs, to get the fastest
    throughput possible. Must be used as a context manager to ensure the GPUs are correctly
    released.

    :param data_source: Contains the vectors/matrices to be synthesized.
    :param network_path: Path to the pickled network to use for synthesis.
    :param num_gpus: The number of GPUs to spread the compute across. If not given, defaults
    to max possible count.
    :return: A context manager for the frame source, which are the outputs from the network in
    order.
    """

    first_data, inputs = iterator_common.first_item_from_iterator(data_source)

    # Default to the max possible GPUs.
    if num_gpus is None:
        num_gpus = len(list(nvsmi.get_gpus()))
        logging.info(f"Found: {num_gpus} GPUs.")

    queue: "Queue[int]" = Queue()
    for item in range(num_gpus):
        queue.put(item)

    output_control_semaphore = Semaphore(1000)

    # Really important here that `processes` always matches the GPU count.
    # Note to future self: do not increase this number to try and go faster.
    with multiprocessing.Pool(
        processes=num_gpus,
        initializer=initializer,
        initargs=(network_path, queue, output_control_semaphore),
    ) as p:

        def locked_output() -> ImageSourceType:
            """
            Releases the semaphore once per frame, to make sure we don't accidentally build up
            back pressure causing a memory leak.
            :return: Outputs from the network in synthesis order.
            """

            for frame in p.imap(partial(synthesize_frame, is_vector(first_data)), inputs):
                yield frame
                output_control_semaphore.release()

        yield locked_output()
