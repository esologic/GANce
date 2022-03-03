"""
Unit tests for the visualization module
"""

import datetime
from typing import Iterator

import more_itertools
import numpy as np
import pytest

import gance.vector_sources.music
import gance.vector_sources.primatives
from gance.data_into_network_visualization.vectors_3d import _reshape_vectors_for_3d_plotting


def reshape_using_chunks(data: np.ndarray, chunk_width: int) -> np.ndarray:
    """
    This is the nieve approach.
    Uses a sampler to split the data into chunks, then manually re-assemble those chunks into a
    list of points (x, y, z).
    This is very slow and inefficient, but it gives us exactly what we want.
    :param data: The data to reshape.
    :param chunk_width: The length of the vector.
    :return: The reshaped data.
    """

    def create_sampler(
        data: np.ndarray, samples_per_chunk: int, reshape: bool = True
    ) -> Iterator[np.ndarray]:
        """
        Create the iterator for the data.
        :param data:  The wav data to turn into chunks.
        :param samples_per_chunk: Data points per sub list (chunk).
        :param reshape: Should be True when feeding into network.
        :return: Yield `data` in chunks.
        """
        for c in more_itertools.chunked(data, n=samples_per_chunk):
            yield np.reshape(np.array(c), (1, len(c))) if reshape else c

    sampler = create_sampler(data, chunk_width, False)

    points = []

    # Very slow and deliberate
    for chunk_index, chunk in enumerate(sampler):
        for point_index, point in enumerate(chunk):
            points.append([point_index, chunk_index, point])

    return np.array(points)


@pytest.mark.parametrize("vector_length,num_vectors", [(100, 3), (1234, 1), (1234, 100)])
def test__reshape_vectors_for_3d_plotting(vector_length: int, num_vectors: int) -> None:
    """
    Reshape some known data using the fast reshape function and compare it to the slow version.
    They should match.
    :param vector_length: The length of the vector, the num points on the x axis.
    :param num_vectors: The number of vectors to generate, this is the y axis on the plots.
    :return: None
    """

    data = gance.vector_sources.primatives.gaussian_data(vector_length, num_vectors)

    good_reshape_start = datetime.datetime.now()
    good_reshape = reshape_using_chunks(data, vector_length)
    good_reshape_end = datetime.datetime.now()

    fast_reshape_start = datetime.datetime.now()
    output = _reshape_vectors_for_3d_plotting(data, vector_length)
    fast_reshape_end = datetime.datetime.now()

    good_reshape_time = (good_reshape_end - good_reshape_start).total_seconds()
    fast_reshape_time = (fast_reshape_end - fast_reshape_start).total_seconds()

    assert np.all(good_reshape == output)
    assert good_reshape_time > fast_reshape_time
