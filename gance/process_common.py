"""
Common functionality related to moving compute to a child process.
"""

from multiprocessing import Process, Queue  # pylint: disable=unused-import
from queue import Empty
from typing import Any, Optional, Union  # pylint: disable=unused-import
from itertools import count
from gance.logger_common import LOGGER

COMPLETE_SENTINEL = "It's Just Wind"


def empty_queue_sentinel(queue: "Queue[Union[str, Any]]", sentinel_count: int = 1) -> None:
    """
    Assumes someone has put the sentinel value into the queue, and continuously pulls items out
    of the queue until it's found. Allows caller to be very sure that the input queue is empty
    and consumers can be `.join`ed.
    :param queue: The queue to empty and search.
    :return: None
    """
    sentinels_found = count(1)
    while True:
        try:
            value = queue.get_nowait()
            if value == COMPLETE_SENTINEL:
                current_count = next(sentinels_found)
                LOGGER.info(f"Found sentinel {current_count}/{sentinel_count} during empty.")
                if current_count == sentinel_count:
                    return None
        except Empty:
            pass


def cleanup_worker(process: Process, timeout: int = 1) -> None:
    """
    Cleans up a worker process that should be ready to be joined.
    If the process can't be joined, we try and send a `.terminate`, then join again.
    If this fails, a `ValueError` is raised.
    :param process: Process to clean up, should already be exited.
    :param timeout: Join timeout.
    :return: None
    :raises: ValueError if the process can't be cleaned up.
    """

    process.join(timeout=timeout)

    if process.exitcode is None:
        LOGGER.warning("Projection worker still alive, sending terminate.")
        process.terminate()
        process.join(timeout=timeout)
        if process.exitcode is None:
            raise ValueError(f"Worker process pid: {process.pid} still running!")
    else:
        LOGGER.debug(f"Worked exited with code: {process.exitcode}")
