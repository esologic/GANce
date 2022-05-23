"""
Functionality for visualizing an iterator of matrices/vectors.
Uses the dark magic of an initialized multiprocessing pool also found in `fast_synthesis`.
"""

import multiprocessing
from functools import partial
from typing import Iterator, Optional, Tuple, Union, cast, overload

import numpy as np
from lz.transposition import transpose

from gance import iterator_common
from gance.data_into_network_visualization import vectors_to_image
from gance.data_into_network_visualization.vectors_to_image import SingleVectorViz
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.image_sources.image_sources_common import ImageResolution
from gance.vector_sources.vector_sources_common import underlying_length
from gance.vector_sources.vector_types import SingleMatrix, SingleVector

# Creates a new one of these in each `multiprocessing.Pool` child thread. That way we can store
# large objects that will also be accessible in the `i/map(func...` call.
_visualizer: Optional[SingleVectorViz] = None


def visualizer_initializer(output_width: int, output_height: int) -> None:
    """
    Sets up the `_visualizer` global variable for the given process.
    Allows children within `imap` to quickly create a frame without setup.
    :param output_width: width of output images.
    :param output_height: height of output images.
    :return: None
    """

    global _visualizer  # pylint: disable=global-statement
    _visualizer = vectors_to_image.vector_visualizer(
        output_width=output_width, output_height=output_height, title=""
    )


def _create_frame(
    x_values: np.ndarray,
    title_prefix: str,
    index_data: Tuple[int, Union[SingleVector, SingleMatrix]],
) -> Tuple[Union[SingleVector, SingleMatrix], RGBInt8ImageType]:
    """
    Creates a visualization image for the given input. Called within a child process.
    :param x_values: x axis for the output image.
    :param title_prefix: Will be prepended to the title, the name of the thing you're visualizing.
    :param index_data: A tuple, the bit of data to visualize, and the index in the source iterator
    it is. The int is consumed in the title of the resulting visualization.
    :return: A tuple, (the data- the vector or matrix that created this frame,
    the visualization image for that input data.)
    """

    # Unpack. Would use a starmap if there was like an `istarmap`
    index, data = index_data

    # Use the global namespace `_visualizer` for the child process that is being called by
    # this function.
    with _visualizer(
        x_values=x_values, y_values=data, new_title=f"{title_prefix} item #{index}"
    ) as visualization:
        return data, cast(RGBInt8ImageType, visualization)


@overload
def visualize_data_source(
    source: Iterator[SingleVector], title_prefix: str, resolution: ImageResolution
) -> Tuple[Iterator[SingleVector], ImageSourceType]:
    ...


@overload
def visualize_data_source(
    source: Iterator[SingleMatrix], title_prefix: str, resolution: ImageResolution
) -> Tuple[Iterator[SingleMatrix], ImageSourceType]:
    ...


def visualize_data_source(
    source: Union[Iterator[SingleVector], Iterator[SingleMatrix]],
    title_prefix: str = "Iterator",
    resolution: ImageResolution = ImageResolution(width=1000, height=1000),
) -> Tuple[Union[Iterator[SingleVector], Iterator[SingleMatrix]], ImageSourceType]:
    """
    Creates a visualization of the data in `source` as an iterator of images. Also forwards
    input data so it can be consumed again.
    Infers vector length from the first item in `source`.
    :param resolution: Resolution of output visualization frames.
    :param source: To visualize.
    :param title_prefix: Will be prepended to the title of the resulting visualization images.
    :return: Tuple of iterators. (The input data, images of visualizations of the input data)
    """

    # Used to set up shape of output
    first, s = iterator_common.first_item_from_iterator(source)

    # only need to create this once.
    x_values = np.arange(underlying_length(cast(np.ndarray, first)))

    def create_output() -> Iterator[Tuple[Union[SingleVector, SingleMatrix], RGBInt8ImageType]]:
        """
        :return: Creates an iterator of tuples of the output.
        """
        with multiprocessing.Pool(
            initializer=visualizer_initializer, initargs=(resolution.width, resolution.height)
        ) as p:
            yield from p.imap(partial(_create_frame, x_values, title_prefix), enumerate(s))

    # Transpose from an iterator of tuples, to a tuple of iterators in the desired output
    # type.
    return cast(
        Tuple[Union[Iterator[SingleVector], Iterator[SingleMatrix]], ImageSourceType],
        transpose(create_output()),
    )
