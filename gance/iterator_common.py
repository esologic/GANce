"""
Common operations on iterators, which are the canonical representation of videos, data sources etc
throughout this application.
"""

import collections
import datetime
import itertools
from typing import Callable, Deque, Iterator, List, Tuple, TypeVar

import more_itertools

from gance.logger_common import LOGGER

T = TypeVar("T")


def first_item_from_iterator(iterator: Iterator[T]) -> Tuple[T, Iterator[T]]:
    """
    Get the first item out of an iterator to surmise some properties of the rest of the items.
    :param iterator: To preview.
    :return: A tuple:
    (the first item in the iterator, the FULL iterator,
    note that the first item is added back onto this output iterator )
    """

    try:
        first_item = next(iterator)
    except StopIteration:
        LOGGER.error("Iterator source was empty, nothing to preview.")
        raise

    return first_item, itertools.chain([first_item], iterator)


def items_per_second(source: Iterator[T]) -> Iterator[T]:
    """
    Prints logging around how often items are being extracted.
    :param source: To forward.
    :return: The input iterator.
    """

    queue_size = 60
    queue_count = itertools.count()
    queue: Deque[datetime.datetime] = collections.deque(maxlen=queue_size)

    def yield_item(item: T) -> T:
        """
        Return the input item, printing speed logging along the way.
        :param item: To forward.
        :return: The input item, unmodified.
        """
        queue.append(datetime.datetime.now())
        if next(queue_count) >= queue_size:
            LOGGER.info(
                f"The last {queue_size} items were consumed at a rate of: "
                f"{queue_size / ((queue[-1] - queue[0]).total_seconds())} items per second."
            )

        # Don't do anything to the input item.
        return item

    return map(yield_item, source)


G = TypeVar("G")


def apply_to_chunk(func: Callable[[List[T]], G], n: int, source: Iterator[T]) -> Iterator[T]:
    """

    :param func:
    :return:
    """

    chunks = more_itertools.chunked(iterable=source, n=n)

    return map(func, chunks)
