"""
Common functionality for working with vectors/arrays of vectors.
Again, at the end of the day, a vector is just a list.

Note: `scale_vectors_to_length_linspace` and `scale_vectors_to_length_resample` work similarly
but can produce deceptively different results. The resample function I think does a better job,
but linspace is more literal.
"""

import math
from typing import Callable, Iterable, Iterator, Tuple, Union, overload

import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.signal import resample, savgol_filter

from gance.vector_sources.vector_types import (
    ConcatenatedMatrices,
    ConcatenatedVectors,
    DividedMatrices,
    DividedVectors,
    SingleMatrix,
    SingleVector,
    is_vector,
)


def pad_array(array: np.ndarray, size: int) -> np.ndarray:
    """
    Pads an array with zeros to the end to a given length.
    :param array: The array to pad.
    :param size: The target length of the array.
    :return: The padded array.
    """
    return np.pad(array, pad_width=(0, size - len(array)), mode="constant")


def remap_values_into_range(
    data: Iterator[Union[float, int]],
    input_range: Tuple[Union[float, int], Union[float, int]],
    output_range: Tuple[Union[float, int], Union[float, int]],
) -> Iterable[float]:
    """
    Scale a list of values within a given bound to a new set of boundaries.
    Uses a multiprocessing Pool to spread the work out across multiple threads.
    :param data: The list of values to scale to the new range.
    :param input_range: The possible (min, max) values of the input range.
    :param output_range: The range the map the new values into, So if an input value was the
    min input value, it would now be the min output value etc. If a value was halfway between
    the input min/max, it would now be halfway between the output min/max.
    :return: The result as an iterator.
    """
    scale = interp1d(list(input_range), list(output_range))
    return list(map(scale, list(data)))


def smooth_vector(vector: SingleVector, window_length: int, polyorder: int) -> SingleVector:
    """
    Scale an individual vector to the output length
    :param vector: The vector to scale.
    :param window_length: An input argument to the savgol filter.
    :param polyorder: An input argument to the savgol filter.
    :return: The output vector will be `vector` stretched or shrank to `output_vector_length`
    samples.
    """
    return SingleVector(savgol_filter(x=vector, window_length=window_length, polyorder=polyorder))


@overload
def sub_vectors(data: ConcatenatedMatrices, vector_length: int) -> DividedMatrices:
    ...


@overload
def sub_vectors(data: ConcatenatedVectors, vector_length: int) -> DividedVectors:
    ...


def sub_vectors(
    data: Union[ConcatenatedMatrices, ConcatenatedVectors], vector_length: int
) -> Union[DividedMatrices, DividedVectors]:
    """
    Spit an array of vectors, so one long list made up of sub-vectors, back into a 2d array
    with those sub-vectors isolated.
    :param data: The data to split.
    :param vector_length: The vector length.
    :return: A 2D array of the vectors.
    """

    if len(data.shape) >= 2:
        num_vectors = int(data.shape[-1] / vector_length)
        return DividedMatrices(np.array(np.split(data, num_vectors, axis=-1)))
    else:
        return DividedVectors(np.reshape(data, (-1, vector_length)))


@overload
def underlying_vector_length(data: SingleVector) -> int:
    ...


@overload
def underlying_vector_length(data: SingleMatrix) -> int:
    ...


def underlying_vector_length(data: Union[SingleVector, SingleMatrix]) -> int:
    """
    Decides what type is on the input, then calculates the vector length accordingly.
    :param data: To evaluate.
    :return: Length of vector, or the length of the vectors within the matrix.
    """

    if is_vector(data):
        return int(data.shape[0])
    else:
        return int(data.shape[1])


def smooth_across_vectors(
    data: ConcatenatedVectors, vector_length: int, window_length: int = 7, polyorder: int = 3
) -> ConcatenatedVectors:
    """
    Smooth an array of vectors. Makes it so one vector is similar to the subsequent vector.
    :param data: The array of vectors to smooth.
    :param vector_length: The length of a single vector.
    :param window_length: A smoothing parameter. See `smooth_vector` for details.
    :param polyorder: A smoothing parameter. See `smooth_vector` for details.
    :return: The smoothed array of vectors.
    """

    # Covert the 1D array to a 2D array of individual vectors
    reshaped = sub_vectors(data, vector_length)

    # Flip the array, now each index (so transposed[0], transposed[1] etc) will be the values
    # at each position in the vector across all vectors. So `transposed[0]` will be the first value
    # across all vectors
    transposed = reshaped.transpose()

    # Apply a smoothing algo to each of these new vectors, making transitions between each vector
    # smoother.
    smoothed = np.array(
        [
            smooth_vector(flipped_vector, window_length=window_length, polyorder=polyorder)
            for flipped_vector in transposed
        ]
    )

    # Flip the array back to the original format, and then flatten it from 2D -> 1D
    return ConcatenatedVectors(smoothed.transpose().flatten())


def smooth_each_vector(
    data: ConcatenatedVectors, vector_length: int, window_length: int = 51, polyorder: int = 2
) -> ConcatenatedVectors:
    """
    Smooth each individual sub-vector with an array of vectors. Subsequent vectors are not smoothed
    together, smoothing only happens within a given sub-vector.
    :param data: The array of vectors to apply smoothing to.
    :param vector_length: The length of the sub vectors.
    :param window_length: Smoothing parameter, see called function docs for more.
    :param polyorder: Smoothing parameter, see called function docs for more.
    :return: The smoothed vectors back in the original shape.
    """
    return ConcatenatedVectors(
        np.concatenate(
            [
                smooth_vector(vector, window_length=window_length, polyorder=polyorder)
                for vector in sub_vectors(data, vector_length)
            ]
        )
    )


def _scale_vectors_with_function(
    data: ConcatenatedVectors,
    original_vector_length: int,
    scale_function: Callable[[SingleVector], SingleVector],
) -> ConcatenatedVectors:
    """
    Scale each vector in an array or vectors using a scaling function.
    :param data: The data to scale.
    :param original_vector_length: The vectors in `data` are all this long.
    :param scale_function: Each vector is passed into this function to get a scaled vector.
    :return: Scaled vectors as an array of vectors.
    """
    return ConcatenatedVectors(
        np.concatenate(
            [scale_function(vector) for vector in sub_vectors(data, original_vector_length)],
            axis=None,
        )
    )


def scale_vectors_to_length_resample(
    data: ConcatenatedVectors, original_vector_length: int, output_vector_length: int
) -> ConcatenatedVectors:
    """
    Given an array of vectors, scale each of those vectors to a new length using linear
    interpolation.
    :param data: The data to scale.
    :param original_vector_length: The vectors in `data` are all this long.
    :param output_vector_length: The vectors in the output will be this long.
    :return: The scaled vectors concatenated together.
    """
    return ConcatenatedVectors(
        _scale_vectors_with_function(
            data=data,
            original_vector_length=original_vector_length,
            scale_function=lambda vector: SingleVector(
                resample(x=vector, num=output_vector_length)
            ),
        )
    )


def scale_vectors_to_length_linspace(
    data: ConcatenatedVectors, original_vector_length: int, output_vector_length: int
) -> ConcatenatedVectors:
    """
    Given an array of vectors, scale each of those vectors to a new length using a 1d linear space
    interpolation.
    Note: I'm not sure if this works correctly, but it has served it's purpose here and there.
    :param data: The data to scale.
    :param original_vector_length: The vectors in `data` are all this long.
    :param output_vector_length: The vectors in the output will be this long.
    :return: The scaled vectors concatenated together.
    """

    input_x_values = np.arange(0, original_vector_length)
    output_x_values = np.linspace(
        0, original_vector_length - 1, num=output_vector_length, endpoint=True
    )

    def scale_vector(vector: SingleVector) -> SingleVector:
        """
        Scale an individual vector to the output length
        :param vector: The vector to scale.
        :return: The output vector will be `vector` stretched or shrank to `output_vector_length`
        samples.
        """
        interp_function = interpolate.interp1d(input_x_values, vector, kind="cubic")
        return SingleVector(interp_function(output_x_values))

    return _scale_vectors_with_function(
        data=data, original_vector_length=original_vector_length, scale_function=scale_vector
    )


def interpolate_to_vector_count(
    data: ConcatenatedVectors, vector_length: int, target_vector_count: int
) -> ConcatenatedVectors:
    """
    Given the individual vectors in the vector array `data`, use `interp1d` to create vectors
    between these vectors until you have `target_vector_count`. You could also use this to reduce
    the number of vectors in `data` but that isn't the target use case of this function.
    :param data: Vector array.
    :param vector_length: Length of a vector in `data`.
    :param target_vector_count: The desired number of vectors of length `vector_length` in the
    output.
    :return: A vector array as described.
    """

    split = sub_vectors(data=data, vector_length=vector_length)

    # Each sub array here is the point in the vector across the set of vectors.
    points_over_time = split.swapaxes(1, 0)

    original_x_points = np.arange(points_over_time.shape[1])

    new_x_points = np.linspace(start=0, stop=max(original_x_points), num=target_vector_count)

    scaled_points_over_time = np.array(
        [interp1d(original_x_points, points)(new_x_points) for points in points_over_time]
    )

    by_vector = np.array(scaled_points_over_time).swapaxes(1, 0)

    return ConcatenatedVectors(np.concatenate(by_vector))


def duplicate_to_vector_count(
    data: ConcatenatedVectors, vector_length: int, target_vector_count: int
) -> ConcatenatedVectors:
    """
    Duplicate sub vectors until the `target_vector_count` is reached in the output.
    Each vector needs to be duplicated the same number of times or a `ValueError` is raised.
    :param data: Vector array.
    :param vector_length: Length of a vector in `data`.
    :param target_vector_count: The desired number of vectors of length `vector_length` in the
    output.
    :return: A vector array as described.
    :raises ValueError: If the number of repeats needed isn't a whole number.
    """

    split = sub_vectors(data=data, vector_length=vector_length)

    # Each sub array here is the point in the vector across the set of vectors.
    points_over_time = split.swapaxes(1, 0)

    frac, repeats = math.modf(target_vector_count / len(split))

    if frac != 0:
        raise ValueError("Cannot repeat values in input to evenly get desired output length")

    scaled_points_over_time = np.array([np.repeat(points, repeats) for points in points_over_time])

    by_vector = np.array(scaled_points_over_time).swapaxes(1, 0)

    return ConcatenatedVectors(np.concatenate(by_vector))


def promote_to_matrix_duplicate(
    data: ConcatenatedVectors, target_depth: int
) -> ConcatenatedMatrices:
    """
    Will be used to take a 1 dimensional vector and promote it to a shape that it can be
    input into full depth model.
    :param data: data to duplicate.
    :param target_depth: The number of times to duplicate `data`. New shape will be
    (`target_depth`, len(data)).
    :return: Duplicated data.
    """

    dimensions = len(data.shape)

    if dimensions != 1:
        raise ValueError("Undefined behavior!")

    return ConcatenatedMatrices(np.tile(data, (target_depth, dimensions)))


@overload
def demote_to_vector_select(data: SingleMatrix, index_to_take: int = 0) -> SingleMatrix:
    ...


@overload
def demote_to_vector_select(
    data: ConcatenatedMatrices, index_to_take: int = 0
) -> ConcatenatedVectors:
    ...


def demote_to_vector_select(
    data: Union[SingleMatrix, ConcatenatedMatrices], index_to_take: int = 0
) -> Union[SingleMatrix, ConcatenatedMatrices]:
    """
    Will be used to take a 1 dimensional vector and promote it to a shape that it can be
    input into full depth model.
    :param data: data to duplicate.
    :param target_depth: The number of times to duplicate `data`. New shape will be
    (`target_depth`, len(data)).
    :return: Duplicated data.
    """
    return ConcatenatedVectors(data[index_to_take])


@overload
def rotate_vectors_over_time(
    data: ConcatenatedVectors, vector_length: int, roll_values: np.ndarray
) -> ConcatenatedVectors:
    ...


@overload
def rotate_vectors_over_time(
    data: ConcatenatedMatrices, vector_length: int, roll_values: np.ndarray
) -> ConcatenatedMatrices:
    ...


def rotate_vectors_over_time(
    data: Union[ConcatenatedVectors, ConcatenatedMatrices],
    vector_length: int,
    roll_values: np.ndarray,
) -> np.ndarray:
    """

    :param data:
    :param vector_length:
    :param roll_values:
    :return:
    """

    split = sub_vectors(data, vector_length)
    roll_per_vector = np.cumsum(roll_values)
    rolled = [
        np.roll(sub_vector, roll_value * -1)
        for sub_vector, roll_value in zip(split, roll_per_vector)
    ]
    return np.concatenate(rolled)
