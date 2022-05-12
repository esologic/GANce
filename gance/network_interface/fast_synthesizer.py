import datetime
import heapq
from contextlib import contextmanager
from itertools import count
from multiprocessing import Event, Queue
from pathlib import Path
from queue import Empty, Full
import logging
from threading import Thread
from multiprocessing import Process
from typing import Dict, Iterator, List, NamedTuple, Union, overload

from gance import iterator_common, process_common
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import (
    _NetworkInput,
    create_network_interface_process,
)
from gance.process_common import COMPLETE_SENTINEL
from gance.vector_sources.vector_sources_common import SingleMatrix, SingleVector, is_vector


class IndexedNetworkInput(NamedTuple):
    """
    Adds an index value to a network input, so the resulting images can be displayed in order.
    """

    index: int
    network_input: Union[SingleMatrix, SingleVector]


class IndexedNetworkOutput(NamedTuple):
    """
    Preserves the input's index, so the resulting images can be displayed in order.
    """

    index: int
    network_output: RGBInt8ImageType


class ThreadStopEvent(NamedTuple):
    """
    Intermediate type
    """

    thread: Thread
    stop_event: Event


@overload
def fast_synthesizer(
    iterator: Iterator[SingleVector], queue: "Queue[Union[IndexedNetworkInput, str]]"
) -> None:
    ...


@overload
def _iterator_into_queue(
    iterator: Iterator[SingleMatrix], queue: "Queue[Union[IndexedNetworkInput, str]]"
) -> None:
    ...


def _iterator_into_queue(
    iterator: Union[Iterator[SingleVector], Iterator[SingleMatrix]],
    queue: "Queue[Union[IndexedNetworkInput, str]]",
    stop_event: Event,
) -> None:
    """

    :param iterator:
    :param queue:
    :return:
    """
    try:
        for index, item in enumerate(iterator):
            if not stop_event.is_set():
                LOGGER.info("Waiting to move item from iterator into queue.")
                queue.put(
                    IndexedNetworkInput(
                        index=index,
                        network_input=item,
                    ),
                )
            else:
                LOGGER.info("Iterator into queue stopped!")
                break
    except BaseException:
        pass

    LOGGER.info("Iterator into queue added complete.")
    queue.put(COMPLETE_SENTINEL)


def _pull_network_input_synthesize_frame(
    network_path: Path,
    gpu_index: int,
    inputs_are_vectors: bool,
    input_queue: "Queue[Union[IndexedNetworkInput, str]]",
    output_queue: "Queue[Union[IndexedNetworkOutput, str]]",
    stop_event: Event,
) -> None:
    """

    :param network_path:
    :param gpu_index:
    :param input_queue:
    :param output_queue:
    :return:
    """

    try:
        network_interface = create_network_interface_process(
        network_path=network_path, gpu_index=gpu_index
        )
    except BaseException as e :
        raise e

    synthesis_function = (
        network_interface.network_interface.create_image_vector
        if inputs_are_vectors
        else network_interface.network_interface.create_image_matrix
    )

    while not stop_event.is_set():
        try:
            indexed_network_input: Union[IndexedNetworkInput, str] = input_queue.get_nowait()

            start = datetime.datetime.now()
            network_output = synthesis_function(indexed_network_input.network_input)
            end = datetime.datetime.now()

            logging.info(f"Synthesis time: {(end - start).total_seconds()}")

            logging.info("Waiting to put input into output queue...")
            output_queue.put(
                IndexedNetworkOutput(
                    index=indexed_network_input.index,
                    network_output=network_output,
                )
            )
            logging.info("Output added.")
        except Empty:
            logging.info("No input to process..")
        except BaseException as e:
            logging.error(f"Ran into exception during loop: {e}")
            break

    logging.info(f"Stopping GPU thread for index: {gpu_index}")
    network_interface.stop_function()

    logging.info("Putting complete in output queue.")
    output_queue.put(COMPLETE_SENTINEL)


@overload
def fast_synthesizer(data_source: Iterator[SingleVector], network_path: Path) -> ImageSourceType:
    ...


@overload
def fast_synthesizer(data_source: Iterator[SingleMatrix], network_path: Path) -> ImageSourceType:
    ...


@contextmanager
def fast_synthesizer(
    data_source: Union[Iterator[SingleVector], Iterator[SingleMatrix]], network_path: Path
) -> Iterator[ImageSourceType]:
    """
    Split the

    :param data_source:
    :param network_path:
    :return:
    """

    input_queue: "Queue[Union[IndexedNetworkInput, str]]" = Queue(maxsize=4)
    output_queue: "Queue[Union[IndexedNetworkOutput, str]]" = Queue(maxsize=20)

    first_data, inputs = iterator_common.first_item_from_iterator(data_source)

    def create_thread_stop_event(target, args) -> ThreadStopEvent:
        """

        :param target:
        :param args:
        :return:
        """

        event = Event()

        return ThreadStopEvent(
            thread=Process(
                target=target,
                args=args + (event,),
            ),
            stop_event=event,
        )

    output_threads, output_stop_events = tuple(
        map(
            list,
            zip(
                *[
                    create_thread_stop_event(
                        target=_pull_network_input_synthesize_frame,
                        args=(
                            network_path,
                            gpu_index,
                            is_vector(first_data),
                            input_queue,
                            output_queue,
                        ),
                    )
                    for gpu_index in range(4)
                ]
            ),
        )
    )

    input_thread, input_stop_event = create_thread_stop_event(
        target=_iterator_into_queue, args=(inputs, input_queue)
    )

    input_thread.start()
    for thread in output_threads:
        thread.start()



    def create_output_iterator() -> ImageSourceType:

        output_buffer: Dict[int, RGBInt8ImageType] = {}

        for index in count():
            for _ in range(10):
                index_network_output: IndexedNetworkOutput = output_queue.get()

                if index_network_output == COMPLETE_SENTINEL:
                    LOGGER.error("Found a complete sentinel from output worker")

                if index_network_output.index == index:
                    output_item = index_network_output
                    yield output_item.network_output
                    break
                else:
                    output_buffer[index_network_output.index] = index_network_output.network_output
                    try:
                        output_item = output_buffer.pop(index)
                        yield output_item
                        break
                    except KeyError:
                        LOGGER.debug(f"Couldn't find {index} in output buffer")
            else:
                LOGGER.warning(f"Never found {index} in output buffer, moving onto next frame.")

    def cleanup() -> None:
        """

        :return:
        """
        LOGGER.info("Setting stop events for output...")
        for stop_event in output_stop_events:
            stop_event.set()

        process_common.empty_queue_sentinel(output_queue, 4)

        LOGGER.info("Cleaning up input")
        input_stop_event.set()
        process_common.empty_queue_sentinel(input_queue)


        LOGGER.info("Queues empty.")

        LOGGER.info("Joining threads...")
        for t in output_threads:
            LOGGER.info(f"joining {t}")
            t.join()
        input_thread.join()
        LOGGER.info("Threads joined.")

    try:
        yield create_output_iterator()
    except Exception:
        pass
    finally:
        cleanup()

    print("here")
    # TODO Add cleanup code the release GPUs
