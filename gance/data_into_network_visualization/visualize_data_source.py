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
    :return: None
    """

    global _visualizer  # pylint: disable=global-statement
    _visualizer = vectors_to_image.vector_visualizer(
        output_width=output_width, output_height=output_height, title=""
    )


def _create_frame(
    x_values, title_prefix, data_index
) -> Tuple[Union[SingleVector, SingleMatrix], RGBInt8ImageType]:
    """

    :return:
    """

    index, data = data_index

    with _visualizer(
        x_values=x_values, y_values=data, new_title=f"{title_prefix} item #{index}"
    ) as visualization:
        return data, visualization


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
    Infers vector length from the first item in `source`.
    :param resolution:
    :param source:
    :param title_prefix:
    :return:
    """

    first, source = iterator_common.first_item_from_iterator(source)

    x_values = np.arange(underlying_length(first))

    def create_output():
        # Really important here that `processes` always matches the GPU count.
        # Note to future self: do not increase this number to try and go faster.
        with multiprocessing.Pool(
            initializer=visualizer_initializer, initargs=(resolution.width, resolution.height)
        ) as p:
            yield from p.imap(partial(_create_frame, x_values, title_prefix), enumerate(source))

    return transpose(create_output())
