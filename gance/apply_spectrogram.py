"""
Functions to convert data to a spectrogram (functionally a series of FFTs) of that data
Based on : https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
"""

from typing import Optional, Tuple

import numpy as np
from skimage import util
from sklearn.preprocessing import minmax_scale

from gance.vector_sources.vector_sources_common import (
    scale_vectors_to_length_resample,
    smooth_across_vectors,
    smooth_each_vector,
)
from gance.vector_sources.vector_types import ConcatenatedVectors


def reshape_spectrogram_to_vectors(
    spectrogram_data: "np.ndarray[np.float32]",
    vector_length: int,
    amplitude_range: Optional[Tuple[int, int]] = None,
) -> ConcatenatedVectors:
    """
    Transpose a given spectrogram output to be the canonical vector shape.
    :param spectrogram_data: The data to reshape, should be output from a spectrogram function.
    :param vector_length:
    :param amplitude_range: Will scale the output amplitude to this range if provided.
    :return: The reshaped data.
    """
    transposed = np.transpose(spectrogram_data)
    flattened = transposed.flatten()
    original_width = transposed.shape[1]
    scaled_to_vector_length = scale_vectors_to_length_resample(
        # This cast is kind of a lie, these are in the shape of
        # `ConcatenatedVectors` but aren't actually Concatenated Vectors.
        data=ConcatenatedVectors(flattened),
        original_vector_length=original_width,
        output_vector_length=vector_length,
    )
    return (
        ConcatenatedVectors(minmax_scale(scaled_to_vector_length, feature_range=amplitude_range))
        if amplitude_range is not None
        else scaled_to_vector_length
    )


def compute_spectrogram(
    data: ConcatenatedVectors, num_frequency_bins: int, truncate: bool = True
) -> np.ndarray:
    """
    Create a spectrogram of a given array.
    :param data: The data to compute the spectrogram of. Needs to be "mono" having a 1-D shape.
    If "stereo" data is fed in it will be converted to "mono".
    :param num_frequency_bins: This can be thought of as the "vector length" or "samples per
    iteration" variables used in many other places in this module.
    :return: The spectrogram as an ndarray. The 0th axis, (`s[0]`, `s[1]`, `s[2]` etc) represents
    the values at a given frequency over time. So for example `output[0][0]` would be the first
    frequency at the first time, and `output[0][-1]` would be the first frequency at the last point
    in time.
    """
    try:
        data = np.mean(data, axis=1)
    except IndexError:
        pass

    m = num_frequency_bins - 1 * 2
    slices = util.view_as_windows(data, window_shape=(m,), step=num_frequency_bins)
    win = np.hanning(m + 1)[:-1]
    slices = slices * win
    slices = slices.T
    fft = np.fft.fft(slices, axis=0)

    if truncate:
        spectrum = fft[: (m // 2)]
    else:
        spectrum = fft

    s = np.abs(spectrum)
    s = 20 * np.log10(s / np.max(s))
    return s


def compute_spectrogram_smooth_scale(
    data: ConcatenatedVectors, vector_length: int, amplitude_range: Optional[Tuple[int, int]] = None
) -> ConcatenatedVectors:
    """
    This function:
    * Parses an array into vectors
    * Computes a spectrogram (a series of FFTs) on these vectors.
    * Smooths these spectrogram vectors so the transition between each of them isn't too dramatic.
    * Scales the output to a given amplitude range.
    :param data: The data compute this analysis on. Even though this might not explicitly be vectors
    as termed in the rest of this application, for this analysis they'll be treated as vectors.
    :param vector_length: The length of the vectors in `data`.
    :param amplitude_range: If given, the output vectors will be scaled to this amplitude.
    :return: The smoothed spectrogram vectors.
    """

    spectrogram = compute_spectrogram(data, vector_length)

    spectrogram_as_vectors = reshape_spectrogram_to_vectors(
        spectrogram, amplitude_range=amplitude_range, vector_length=vector_length
    )

    smoothed_spectrogram = smooth_across_vectors(
        spectrogram_as_vectors,
        vector_length,
        window_length=7,
        polyorder=3,
    )

    each_vector_smoothed = smooth_each_vector(
        data=smoothed_spectrogram, vector_length=vector_length, window_length=5, polyorder=3
    )

    return each_vector_smoothed
