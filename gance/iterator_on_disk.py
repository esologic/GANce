"""
Thank u: https://stackoverflow.com/questions/21157739/how-to-iterate-through-a-python-queue-queue-with-a-for-loop-instead-of-a-while-l
"""

import pickle
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Tuple, TypeVar

IterationResult = TypeVar("IterationResult")

from sentinels import NOTHING


def cache_iterator_on_disk(
    iterator: Iterator[IterationResult], copies: int
) -> Tuple[Iterator[IterationResult], Tuple[Iterator[IterationResult], ...]]:
    """

    :param iterator:
    :param copies:
    :return:
    """

    path_queues = [Queue() for _ in range(copies)]

    def item_into_queues(item: Any) -> None:
        """

        :param item:
        :return:
        """
        for queue in path_queues:
            queue.put(item)

    item_into_queues(NOTHING)

    def forward_iterator() -> Iterator[IterationResult]:
        """

        :return:
        """

        for result in iterator:
            # These will get deleted after being loaded into memory later.
            with NamedTemporaryFile(mode="wb", delete=False) as p:
                pickle.dump(result, p)
                item_into_queues(Path(p.name))

            yield result

    def iterate_over_queue(queue: "Queue") -> Iterator[IterationResult]:
        """

        :param queue:
        :return:
        """

        for result_path in iter(queue.get, NOTHING):
            with open(str(result_path), "rb") as p:
                output = pickle.load(p)
            result_path.unlink()
            yield output

    return forward_iterator(), tuple(iterate_over_queue(queue) for queue in path_queues)
