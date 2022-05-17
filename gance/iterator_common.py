"""
Common operations on iterators, which are the canonical representation of videos, data sources etc
throughout this application.
"""

import itertools
from typing import Iterator, Tuple, TypeVar

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
