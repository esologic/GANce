"""
Test functionality in the vectors.common file.
Each test here will probably have some crossover with other functions.
This is good for coverage but bad for side-effects.
"""

from typing import Tuple, cast

import numpy as np
import pytest

from gance.vector_sources import vector_sources_common
from gance.vector_sources.vector_types import ConcatenatedVectors, is_vector


@pytest.mark.parametrize(
    "original_vector_length,num_original_vectors,output_vector_length",
    [(10, 2, 50), (10, 1, 1000)],
)
def test_scale_vectors_to_length(
    original_vector_length: int, num_original_vectors: int, output_vector_length: int
) -> None:
    """
    Test to make sure that the rough qualities of the scaling function works.
    Not an exhaustive test by any means, more of a sanity check.
    :param original_vector_length: Source vector length.
    :param num_original_vectors: Num source vectors in array.
    :param output_vector_length: Length to scale each of the output vectors to.
    :return: None
    """

    len_original = original_vector_length * num_original_vectors

    unscaled = cast(
        ConcatenatedVectors, np.sin(np.linspace(start=0, stop=len_original - 1, num=len_original))
    )

    scaled = vector_sources_common.scale_vectors_to_length_resample(
        data=unscaled,
        original_vector_length=original_vector_length,
        output_vector_length=output_vector_length,
    )

    indexer = np.arange(
        start=0,
        stop=(output_vector_length * num_original_vectors),
        step=output_vector_length / original_vector_length,
    )

    at_points = np.array(
        [
            scaled[scaled_value_index]  # pylint:disable=unsubscriptable-object
            for scaled_value_index in indexer
        ]
    )

    # Pretty forgiving...
    assert np.allclose(unscaled, at_points, atol=0.5)

    assert len(vector_sources_common.sub_vectors(unscaled, original_vector_length)) == len(
        vector_sources_common.sub_vectors(scaled, output_vector_length)
    )

    assert np.isclose(max(unscaled), max(scaled), atol=0.1)
    assert np.isclose(min(unscaled), min(scaled), atol=0.1)


@pytest.mark.parametrize(
    "input_shape,vector_length,expected_output_shape",
    [((1, 10), 5, (2, 1, 5)), ((20, 10), 5, (2, 20, 5)), ((18, 512 * 10), 512, (10, 18, 512))],
)
def test_sub_vectors_shapes(
    input_shape: Tuple[int, ...], vector_length: int, expected_output_shape: Tuple[int, ...]
) -> None:
    """
    Looking at the shape only of the output tells us a lot of about if the function is working as
    expected or not. This + a short test to check content is sufficient to test function.
    :return: None
    """

    data = ConcatenatedVectors(np.zeros(input_shape))
    assert (
        vector_sources_common.sub_vectors(data=data, vector_length=vector_length).shape
        == expected_output_shape
    )


@pytest.mark.parametrize(
    "input_shape,expected_result",
    [((10,), True), ((512,), True), ((1, 10), False), ((18, 512), False)],
)
def test_is_vector(input_shape: Tuple[int, ...], expected_result: bool) -> None:
    """
    Sanity check of this function.
    :param input_shape: Shape of the input data.
    :param expected_result: Expected output of function.
    :return: None
    """
    assert is_vector(np.zeros(shape=input_shape)) == expected_result
