"""
Tries to do `itertools.tee`, but on disk instead of in memory.

Thank u: https://stackoverflow.com/a/70917416
"""

import pickle
import shutil
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, List, NamedTuple, Tuple

import h5py
import numpy as np
from sentinels import NOTHING
from typing_extensions import Protocol

from gance.image_sources.image_sources_common import RGBInt8ImageType


class SerializeItem(Protocol):
    """
    Describes a function that writes a given item out to disk.
    """

    def __call__(self, path: Path, item: Any) -> None:
        """
        :param path: Path to write the serialized object to on disk.
        :param item: Object to serialize.
        :return: None
        """


class DeSerializeItem(Protocol):
    """
    Describes a function that loads an item from disk back into memory.
    """

    def __call__(self, path: Path) -> Any:
        """
        :param path: Path to the object on disk.
        :return: Item loaded back into memory.
        """


class Serializer(NamedTuple):
    """
    A pair of functions, one to write and one to load items from disk.
    """

    serialize: SerializeItem
    deserialize: DeSerializeItem


def serialize_pickle(path: Path, item: Any) -> None:
    """
    Writes an item to disk using the built-in pickle module.
    :param path: Path to write the serialized object to on disk.
    :param item: Object to serialize.
    :return: None
    """

    with open(str(path), "wb") as p:
        pickle.dump(item, p)


def deserialize_pickle(path: Path) -> Any:
    """
    Loads a pickled item from disk using the built-in pickle module.
    :param path: Path to the object on disk.
    :return: Item loaded back into memory.
    """

    with open(str(path), "rb") as p:
        return pickle.load(p)


PICKLE_SERIALIZER = Serializer(serialize=serialize_pickle, deserialize=deserialize_pickle)

HDF5_DATASET_NAME = "item_dataset"


def serialize_hdf5(path: Path, item: RGBInt8ImageType) -> None:
    """
    Writes an item to disk using hdf5, a format for storing data arrays on disk.
    :param path: Path to write the serialized object to on disk.
    :param item: Object to serialize.
    :return: None
    """

    with h5py.File(name=str(path), mode="w") as f:
        f.create_dataset(
            HDF5_DATASET_NAME,
            shape=item.shape,
            dtype=item.dtype,
            data=item,
            compression="gzip",
            shuffle=True,
        )


def deserialize_hdf5(path: Path) -> RGBInt8ImageType:
    """
    Loads an item to disk using hdf5, a format for storing data arrays on disk.
    :param path: Path to the object on disk.
    :return: Item loaded back into memory.
    """

    with h5py.File(name=str(path), mode="r") as f:
        return RGBInt8ImageType(np.array(f[HDF5_DATASET_NAME]))


HDF5_SERIALIZER = Serializer(serialize=serialize_hdf5, deserialize=deserialize_hdf5)


def load_queue_items(queue: "Queue[Path]", deserialize: DeSerializeItem) -> Iterator[Any]:
    """
    Iterate over the items in a queue.
    Load the objects on disk back into memory and yield them.
    Before yielding the objects, deletes their source file.
    :param queue: To consume.
    :param deserialize: Function to load the items from disk.
    :return: An iterator of the items stored in the queue.
    """

    for path in iter(queue.get, NOTHING):
        output: Any = deserialize(path)
        path.unlink()
        yield output


def iterator_on_disk(
    iterator: Iterator[Any],
    copies: int,
    serializer: Serializer = PICKLE_SERIALIZER,
) -> Tuple[Iterator[Any], ...]:
    """
    Caches the results from an input iterator onto disk rather than into memory for re-iteration
    later. Kind of like `itertools.tee`, but instead of going into memory with the copies, the
    intermediate objects are stored on disk.
    :param iterator: The iterator to duplicate.
    :param copies: The number of secondary iterators to make. Think of this like the `n` argument
    to `itertools.tee`.
    :param serializer: Defines how the objects will be stored on disk.
    :return: A tuple:
        (
            The primary iterator. Consume this one to populate the values in the secondary
            iterators.,
            The secondary iterators. When one of these is incremented, its next object
            is loaded from disk and yielded. Note that if you iterate on these past the head of
            `primary`, then the iteration will block.
        )
    """

    path_queues: List["Queue[Path]"] = [Queue() for _ in range(copies)]

    def forward_iterator() -> Iterator[Any]:
        """
        Works through the input iterator, and as new times are produced, saves
        them to disk, and fills the queues with their locations.
        :return: Yields the original items from the input iterator.
        """

        for item in iterator:

            # These will get deleted after being loaded into memory later.
            with NamedTemporaryFile(mode="wb", delete=True) as primary_dump:

                primary_path = Path(primary_dump.name)
                serializer.serialize(path=primary_path, item=item)

                for queue in path_queues:
                    with NamedTemporaryFile(mode="wb", delete=False) as secondary_dump:
                        secondary_path = Path(secondary_dump.name)
                        shutil.copy(src=primary_path, dst=secondary_path)
                        queue.put(secondary_path)

            yield item

        # Tells the queues that no more items will be coming out.
        for queue in path_queues:
            queue.put(NOTHING)

    return (forward_iterator(),) + tuple(
        load_queue_items(queue, deserialize=serializer.deserialize) for queue in path_queues
    )
