"""
Tries to do `itertools.tee`, but on disk instead of in memory.

Thank u: https://stackoverflow.com/a/70917416
"""

import pickle
import shutil
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Iterator, List, Tuple, TypeVar

from sentinels import NOTHING

IterationItem = TypeVar("IterationItem")


def load_queue_items(queue: "Queue[Path]") -> Iterator[IterationItem]:
    """
    Iterate over the items in a queue.
    Load the objects on disk back into memory and yield them.
    Before yielding the objects, deletes their source file.
    :param queue: To consume.
    :return: An iterator of the items stored in the queue.
    """

    def load_item(path: Path) -> IterationItem:
        """
        Helper function.
        :param path: Load the object at this path.
        :return: The loaded object
        """

        with open(str(path), "rb") as p:
            output: IterationItem = pickle.load(p)
            path.unlink()

        return output

    return map(load_item, iter(queue.get, NOTHING))


def iterator_on_disk(
    iterator: Iterator[IterationItem], copies: int
) -> Tuple[Iterator[IterationItem], Tuple[Iterator[IterationItem], ...]]:
    """
    Caches the results from an input iterator onto disk rather than into memory for re-iteration
    later. Kind of like `itertools.tee`, but instead of going into memory with the copies, the
    intermediate objects are stored on disk.
    :param iterator: The iterator to duplicate.
    :param copies: The number of secondary iterators to make. Think of this like the `n` argument
    to `itertools.tee`.
    :return: A tuple:
        (
            The primary iterator. Consume this one to populate the values in the secondary
            iterators.,
            A tuple of secondary iterators. When one of these is incremented, it's next object
            is loaded from disk and yielded. Note that if you iterate on these past the head of
            `primary`, then the iteration will block.
        )
    """

    path_queues: List["Queue[Path]"] = [Queue() for _ in range(copies)]

    def forward_iterator() -> Iterator[  # pylint: disable=inconsistent-return-statements
        IterationItem
    ]:
        """
        Works through the input iterator, and as new times are produced, saves
        them to disk, and fills the queues with their locations.
        :return: Yields the original items from the input iterator.
        """

        for item in iterator:

            # These will get deleted after being loaded into memory later.
            with NamedTemporaryFile(mode="wb", delete=False) as primary_dump:
                pickle.dump(item, primary_dump)

            queues = iter(path_queues)

            try:
                queue = next(queues)
            except StopIteration:
                return None

            queue.put(Path(primary_dump.name))

            for queue in queues:
                with NamedTemporaryFile(mode="wb", delete=False) as secondary_dump:
                    queue.put(Path(shutil.copy(src=primary_dump.name, dst=secondary_dump.name)))

            yield item

        # Tells the queues that no more items will be coming out.
        for queue in path_queues:
            queue.put(NOTHING)

    return forward_iterator(), tuple(load_queue_items(queue) for queue in path_queues)
