"""
Test of the reduction primitives and underlying functions.
"""

from typing import Any, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from gance.vector_sources import vector_reduction

SAMPLE_DATA = np.array([0, 0, 0, 5, 10, 10, 10, 5, 0, 0, 0, 10, 11, 12, 11, 10, 5, 0, 0])


@pytest.mark.parametrize(
    "threshold_value,track_length,expected_result",
    [
        (
            4.0,
            6,
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
        ),
        (
            4.0,
            3,
            [
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
            ],
        ),
        (
            10.0,
            4,
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        ),
        (
            10.0,
            3,
            [
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        ),
    ],
)
def test_threshold_track_filter(
    threshold_value: float, track_length: int, expected_result: List[bool]
) -> None:
    """
    Test to make sure that thresholding and track length filtering works.
    :param threshold_value: Input arg.
    :param track_length: Input arg.
    :param expected_result: Expected output.
    :return: None
    """

    result = vector_reduction.track_length_filter(
        bool_tracks=pd.Series(SAMPLE_DATA) >= threshold_value,
        track_length=track_length,
    )

    assert result == expected_result


@pytest.mark.parametrize(
    "data,expected_value",
    [
        (np.arange(0, 10, 1), 1.0),
        (np.arange(0, 10, 2), 2.0),
        (
            np.array([np.nan for _ in range(10)]),
            0.0,
        ),
    ],
)
def test__derive_data_constant_result(data: npt.NDArray[Any], expected_value: float) -> None:
    """
    Input some curves with constant slopes to make sure the derivation function works as expected.
    Also validate that the none inputs are correctly converted.
    :param data: Input, to derive.
    :param expected_value:
    :return: None
    """

    assert all(
        (
            np.isclose(value, expected_value)
            for value in vector_reduction._derive_data(  # pylint: disable=protected-access
                data=data, order=1
            )
        )
    )
