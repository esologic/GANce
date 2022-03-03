"""
Functions to aide in selecting the model for a given frame in a visualization based on the audio
associated with that frame.
"""

import zlib
from multiprocessing import Pool
from typing import List

import librosa
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter

from gance.data_into_model_visualization.visualization_common import DataLabel, ResultLayers
from gance.vector_sources.vector_sources_common import remap_values_into_range, sub_vectors
from gance.vector_sources.vector_types import ConcatenatedVectors, SingleVector


def _compute_raw_rms(
    time_series_audio_vectors: ConcatenatedVectors, vector_length: int
) -> np.ndarray:
    """
    Helper function. This produces output in the expected shape.
    Where one frame's worth of audio (in the video) is reduced to a single RMS value.
    :param time_series_audio_vectors: The vectors to reduce
    :param vector_length: The number of points in `time_series_audio_vectors` that make up a frame
    in the output video.
    :return: The RMS values.
    """
    return librosa.feature.rms(
        y=time_series_audio_vectors, frame_length=vector_length, center=False
    )[0]


def reduce_vector_rms_rolling_max(
    time_series_audio_vectors: ConcatenatedVectors, vector_length: int
) -> ResultLayers:
    """
    Takes a single time series audio vector and reduces it to its RMS value over that vector.
    :param time_series_audio_vectors: The vectors to reduce.
    :param vector_length: The number of points in `time_series_audio_vectors` that make up a frame
    in the output video.
    :return: The RMS power value as a float. See the library function `librosa.feature.rms` for
    more explanation.
    """

    raw_rms = _compute_raw_rms(time_series_audio_vectors, vector_length)
    feature_length = int(len(raw_rms) / 80)
    output = (  # pylint: disable=unused-variable
        maximum_filter1d(input=raw_rms, size=feature_length) if feature_length > 0 else raw_rms
    )
    return ResultLayers(
        result=DataLabel(output, "Rolling Max"),
        layers=[DataLabel(raw_rms, "Raw RMS Power")],
    )


def _smoothed_rolling_average(
    input_values: DataLabel,
    rolling_average_window: int = 3,
    savgol_window_length: int = 7,
    savgol_polyorder: int = 3,
) -> ResultLayers:
    """
    Computes a rolling average on an input signal, and then smooths the rolling average with a
    savgol filter.
    :param input_values: Contains the values to average/smooth
    :param rolling_average_window: The width of the rolling window, this many points are averaged
    together.
    :param savgol_window_length: See docs in `savgol_filter`.
    :param savgol_polyorder: See docs in `savgol_filter`.
    :return: Assumes this is the output step, and returns a `ResultsLayers` NT with the process.
    """

    as_series = pd.Series(input_values.data)

    # The `.mean()` will contain `nan` values for the first 10 places.
    rolling_average = (
        as_series.rolling(rolling_average_window).mean().fillna(as_series.mean()).to_numpy()
    )

    smoothed_average = savgol_filter(
        x=rolling_average, window_length=savgol_window_length, polyorder=savgol_polyorder
    )

    return ResultLayers(
        result=DataLabel(
            smoothed_average,
            "Savgol Smoothing Filter "
            f"(window={savgol_window_length}, polyorder={savgol_polyorder})",
        ),
        layers=[
            DataLabel(rolling_average, f"Rolling Average (window={rolling_average_window})"),
            input_values,
        ],
    )


def reduce_vector_rms_rolling_average(
    time_series_audio_vectors: ConcatenatedVectors,
    vector_length: int,
    rolling_average_window: int = 3,
    savgol_window_length: int = 7,
    savgol_polyorder: int = 3,
) -> ResultLayers:
    """
    Takes a single time series audio vector and reduces it to its RMS value over that vector.
    :param time_series_audio_vectors: The vectors to reduce.
    :param vector_length: The number of points in `time_series_audio_vectors` that make up a frame
    in the output video.
    :param savgol_window_length: Passed into next call, see those docs.
    :param savgol_polyorder: Passed into next call, see those docs.
    :return: The RMS power value as a float. See the library function `librosa.feature.rms` for
    more explanation.
    """
    return _smoothed_rolling_average(
        DataLabel(_compute_raw_rms(time_series_audio_vectors, vector_length), "Raw RMS Power"),
        rolling_average_window=rolling_average_window,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )


def _compressed_vector_size(vector: SingleVector) -> int:
    """
    Compress a vector, and return the number of bytes in the compressed vectors.
    :param vector: Vector to compress.
    :return: Length of the resulting compressed bytes.
    """
    vector_as_bytes = vector.tobytes()
    compressed_bytes: bytes = zlib.compress(vector_as_bytes)
    return len(compressed_bytes)


def reduce_vector_gzip_compression_rolling_average(
    time_series_audio_vectors: ConcatenatedVectors, vector_length: int
) -> ResultLayers:
    """
    Takes a single time series audio vector and reduces it to its RMS value over that vector.
    :param time_series_audio_vectors: The vectors to reduce.
    :param vector_length: The number of points in `time_series_audio_vectors` that make up a frame
    in the output video.
    :return: The RMS power value as a float. See the library function `librosa.feature.rms` for
    more explanation.
    """

    with Pool() as p:
        compressed_sizes = p.map(
            _compressed_vector_size,
            sub_vectors(data=time_series_audio_vectors, vector_length=vector_length),
        )

    output = DataLabel(np.array(compressed_sizes), "Gzipped Audio")

    return _smoothed_rolling_average(output)


def quantize_results_layers(
    results_layers: ResultLayers,
    model_indices: List[int],
) -> ResultLayers:
    """
    Takes the output of `reducer(
    time_series_audio_vectors=time_series_audio_vectors, vector_length=vector_length)` and:
        * Scales the values into the range of the possible indices.
        * Quantizes the floats from the scaling operation into indexes to be consumed.

    The resulting indexes are used to select which model gets used to create a given frame.
    It's the responsibility of the function given as `reducer` to go from audio -> index.

    :param time_series_audio_vectors: The audio to get reduced into indexes.
    :param vector_length: Each frame in the resulting video will be displayed for this many
    points of audio.
    :param reducer: See docs in the Protocol.
    :param model_indices: The candidate indices to choose from.
    :return: An iterator of the indices. First frame of the generated video (based on the audio)
    should map to first item out of the iterator etc.
    """

    scaled_into_index_range = remap_values_into_range(
        data=results_layers.result.data,
        input_range=(min(results_layers.result.data), max(results_layers.result.data)),
        output_range=(0, len(model_indices) - 1),
    )

    quantized = np.rint(scaled_into_index_range).astype(int)

    return ResultLayers(
        result=DataLabel(quantized, f"{results_layers.result.label} Scaled, Quantized"),
        layers=[results_layers.result] + results_layers.layers,
    )


def _derive_data(data: np.ndarray, order: int) -> np.ndarray:
    """
    Take a given order derivative of the input data.
    Note: `np.nan` values within the input are converted to zero before taking the derivative.
    :param data: To derive.
    :return: derivative as an array, same length as input array.
    """

    data = np.nan_to_num(data)
    x_axis = np.arange(len(data))
    return UnivariateSpline(x=x_axis, y=data).derivative(n=order)(x_axis)


def derive_results_layers(results_layers: ResultLayers, order: int) -> ResultLayers:
    """
    Take a derivative of the results in a `ResultLayers`, add this compute onto the layers.
    :param results_layers: To derive.
    :param order: nth order derivation.
    :return: New NT with this compute added.
    """

    return ResultLayers(
        result=DataLabel(
            _derive_data(data=results_layers.result.data, order=order),
            f"Derevation order={order}",
        ),
        layers=[results_layers.result] + results_layers.layers,
    )


def absolute_value_results_layers(results_layers: ResultLayers) -> ResultLayers:
    """
    Compute the absolute value of the results, add compute onto layers.
    :param results_layers: To take abs value of.
    :return: NT with this new compute added.
    """

    return ResultLayers(
        result=DataLabel(
            np.abs(results_layers.result.data),
            "Absolute Value",
        ),
        layers=[results_layers.result] + results_layers.layers,
    )


def rolling_sum_results_layers(results_layers: ResultLayers, window_length: int) -> ResultLayers:
    """
    Create new result that is a rolling sum of previous results.
    :param results_layers: To sum.
    :return: NT with this new compute added.
    """

    series = pd.Series(results_layers.result.data)

    return ResultLayers(
        result=DataLabel(
            np.array(series.rolling(window_length).sum()),
            f"Rolling Sum (window={window_length})",
        ),
        layers=[results_layers.result] + results_layers.layers,
    )


def track_length_filter(bool_tracks: pd.Series, track_length: int) -> np.ndarray:
    """
    For a list of booleans, reject periods of true that are less than the given length.
    :param bool_tracks: INCLUSIVE!
    :param track_length: INCLUSIVE!
    :return: The input tracks with the shorter tracks replaced with all `False`.
    """

    df = pd.DataFrame({"bool_tracks": bool_tracks.astype(int)})
    df["track_number"] = df.bool_tracks.astype(int).diff(1).fillna(0).abs().cumsum().squeeze()
    df["track_length"] = df.track_number.groupby(df.track_number).transform(len)
    df["output_mask"] = (df.bool_tracks == 1) & (df.track_length >= track_length)

    return df.output_mask.tolist()
