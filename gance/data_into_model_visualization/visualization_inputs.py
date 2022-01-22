"""
Functions that produce `VisualizationInput`s, which are the input to module visualizations.
These are basically transformations of time series audio vectors into things like FFTs etc.
Functions themselves match a standard format so they can be used interchangeably.
"""

from itertools import chain, repeat
from pathlib import Path
from typing import Any, Iterable, List, NamedTuple, Tuple, cast

import numpy as np
from sklearn.preprocessing import minmax_scale

from gance.apply_spectrogram import compute_spectrogram_smooth_scale
from gance.data_into_model_visualization.visualization_common import (
    ResultLayers,
    VisualizationInput,
)
from gance.dynamic_model_switching import model_index_selector, reduce_vector_rms_rolling_average
from gance.gance_types import ImageSourceType
from gance.projection import projection_file_reader
from gance.vector_sources import vector_sources_common
from gance.vector_sources.primatives import Sigmas, gaussian_data
from gance.vector_sources.vector_types import (
    ConcatenatedMatrices,
    ConcatenatedVectors,
    MatricesLabel,
    VectorsLabel,
)


def _create_spectrogram(
    time_series_audio_vectors: ConcatenatedVectors,
    vector_length: int,
    fft_amplitude_range: Tuple[int, int],
    fft_roll_enabled: bool,
) -> ConcatenatedVectors:
    """
    Creates spectrogram based on user config.
    :param time_series_audio_vectors: See docs in the protocol.
    :param vector_length: Canonical def.
    :param fft_amplitude_range: Values in FFT are scaled to this range.
    :param fft_roll_enabled: If true, the FFT vectors move over time.
    :return: FFT spectrogram as vectors.
    """

    spectrogram = compute_spectrogram_smooth_scale(
        data=time_series_audio_vectors,
        vector_length=vector_length,
        amplitude_range=fft_amplitude_range,
    )

    if fft_roll_enabled:
        roll_values: ResultLayers = model_index_selector(
            time_series_audio_vectors=time_series_audio_vectors,
            vector_length=vector_length,
            reducer=reduce_vector_rms_rolling_average,
            model_indices=list(np.arange(0, 3)),
        )

        spectrogram = vector_sources_common.smooth_each_vector(
            data=vector_sources_common.rotate_vectors_over_time(
                data=spectrogram,
                vector_length=vector_length,
                roll_values=roll_values.result.data,
            ),
            vector_length=vector_length,
        )

    return spectrogram


def repeat_each(iterable: Iterable[Any], n: int = 2) -> Iterable[Any]:
    """Repeat each element in *iterable* *n* times.

    >>> list(repeat_each('ABC', 3))
    ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    """
    return chain.from_iterable(map(repeat, iterable, repeat(n)))


def alpha_blend_vectors_max_rms_power_audio(
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    time_series_audio_vectors: np.ndarray,
    vector_length: int,
    model_indices: List[int],
) -> VisualizationInput:
    """
    For vectors:
        * Converts audio to spectrogram.
        * Applies smoothing functions to spectrogram.
        * Creates a gaussian smoothed noise source.
        * Uses alpha blending to combine these two vector sources.

    For index:
        * A rolling max rms power is computed on audio
        * This value is scaled to the range of model indices and quantized to actually select
        the models.

    :param alpha: 0 means no music will be visible in combined signal, 1 means combined signal
    will be entirely music.
    :param fft_roll_enabled: If true, the FFT vectors move over time.
    :param fft_amplitude_range: Values in FFT are scaled to this range.
    :param fft_depth: Number of vectors within the final latents matrices that receive the
    FFT during alpha blending.
    :param time_series_audio_vectors: See docs in the protocol.
    :param vector_length: See docs in the protocol.
    :param model_indices: See docs in the protocol.
    :return: The vector sources to be passed to the visualization functions.
    """

    spectrogram = _create_spectrogram(
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        fft_amplitude_range=fft_amplitude_range,
        fft_roll_enabled=fft_roll_enabled,
    )

    num_vectors = int(spectrogram.shape[0] / vector_length)

    noise = minmax_scale(
        gaussian_data(
            vector_length=vector_length,
            num_vectors=num_vectors,
            sigmas=Sigmas(across_vectors=50, within_vectors=0),
        ),
        feature_range=(-4, 4),
    )

    combined = noise * (1.0 - alpha) + spectrogram * alpha

    indices_layers: ResultLayers = model_index_selector(
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        reducer=reduce_vector_rms_rolling_average,
        model_indices=model_indices,
    )

    return VisualizationInput(
        a_vectors=VectorsLabel(
            data=spectrogram, vector_length=vector_length, label="Audio Spectrogram"
        ),
        b_vectors=VectorsLabel(
            data=noise, vector_length=vector_length, label="Gaussian Smoothed Noise"
        ),
        combined=VectorsLabel(
            data=combined,
            vector_length=vector_length,
            label=f"Combined w/ Alpha Blending, a={alpha}",
        ),
        model_indices=indices_layers.result,
        model_index_layers=indices_layers.layers,
    )


class Output(NamedTuple):

    targets: ImageSourceType
    finals: ImageSourceType
    visualization_input: VisualizationInput


def benis(  # pylint: disable=too-many-locals
    projection_file_path: Path,
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    blend_depth: int,
    time_series_audio_vectors: ConcatenatedVectors,
    vector_length: int,
    model_indices: List[int],
) -> Output:
    """
    For vectors:
        Audio:
            * Converts audio to spectrogram.
            * Applies smoothing functions to spectrogram.
            * Uses RMS power to rotate the spectrogram over time if desired.

        Projection File:
            * Final latents are loaded from file (these are matrices)
            * Final latents are duplicated to fill the desired number of vectors given the
            length of the time series audio vectors

        * Uses alpha blending to combine these two vector sources.

    For index:
        * A rolling max rms power is computed on audio
        * This value is scaled to the range of model indices and quantized to actually select
        the models.

    :param projection_file_path: Path to the projection file on disk.
    :param alpha: 0 means no music will be visible in combined signal, 1 means combined signal
    will be entirely music.
    :param fft_roll_enabled: If true, the FFT vectors move over time.
    :param fft_amplitude_range: Values in FFT are scaled to this range.
    :param blend_depth: Number of vectors within the final latents matrices that receive the
    FFT during alpha blending.
    :param time_series_audio_vectors: See docs in the protocol.
    :param vector_length: See docs in the protocol.
    :param model_indices: See docs in the protocol.
    :return: The vector sources to be passed to the visualization functions.
    :return:
    """

    reader = projection_file_reader.load_projection_file(projection_file_path)

    final_latents = projection_file_reader.final_latents_matrices_label(reader)

    spectrogram = _create_spectrogram(
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        fft_amplitude_range=fft_amplitude_range,
        fft_roll_enabled=fft_roll_enabled,
    )

    num_vectors = int(spectrogram.shape[0] / vector_length)

    duplicated = vector_sources_common.duplicate_to_vector_count(
        # TODO: This is a shortcut that we can take because we know the vectors within the
        # matrix are identical.
        data=vector_sources_common.demote_to_vector_select(final_latents.data, index_to_take=0),
        vector_length=vector_length,
        target_vector_count=num_vectors,
    )

    projected_vectors: ConcatenatedMatrices = vector_sources_common.promote_to_matrix_duplicate(
        data=duplicated.vectors,
        target_depth=final_latents.data.shape[0],
    )

    alpha_blended = vector_sources_common.promote_to_matrix_duplicate(
        ConcatenatedVectors(
            vector_sources_common.demote_to_vector_select(projected_vectors, 0) * (1.0 - alpha)
            + spectrogram * alpha
        ),
        blend_depth,
    )

    combined = np.concatenate(
        (alpha_blended, projected_vectors[blend_depth:18])  # pylint: disable=unsubscriptable-object
    )

    indices_layers: ResultLayers = model_index_selector(
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        reducer=reduce_vector_rms_rolling_average,
        model_indices=model_indices,
    )

    return Output(
        targets=cast(
            ImageSourceType, repeat_each(reader.target_images, duplicated.duplication_factor)
        ),
        finals=cast(
            ImageSourceType, repeat_each(reader.final_images, duplicated.duplication_factor)
        ),
        visualization_input=VisualizationInput(
            a_vectors=VectorsLabel(
                data=spectrogram, vector_length=vector_length, label="Rolled Audio Spectrogram"
            ),
            b_vectors=MatricesLabel(
                data=projected_vectors,
                vector_length=vector_length,
                label=final_latents.label,
            ),
            combined=MatricesLabel(
                data=combined,
                vector_length=vector_length,
                label=f"Combined w/ Alpha Blending, a={alpha}",
            ),
            model_indices=indices_layers.result,
            model_index_layers=indices_layers.layers,
        ),
    )


def alpha_blend_projection_file(  # pylint: disable=too-many-locals
    projection_file_path: Path,
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    blend_depth: int,
    time_series_audio_vectors: ConcatenatedVectors,
    vector_length: int,
    model_indices: List[int],
) -> VisualizationInput:
    """
    For vectors:
        Audio:
            * Converts audio to spectrogram.
            * Applies smoothing functions to spectrogram.
            * Uses RMS power to rotate the spectrogram over time if desired.

        Projection File:
            * Final latents are loaded from file (these are matrices)
            * Final latents are duplicated to fill the desired number of vectors given the
            length of the time series audio vectors

        * Uses alpha blending to combine these two vector sources.

    For index:
        * A rolling max rms power is computed on audio
        * This value is scaled to the range of model indices and quantized to actually select
        the models.

    :param projection_file_path: Path to the projection file on disk.
    :param alpha: 0 means no music will be visible in combined signal, 1 means combined signal
    will be entirely music.
    :param fft_roll_enabled: If true, the FFT vectors move over time.
    :param fft_amplitude_range: Values in FFT are scaled to this range.
    :param blend_depth: Number of vectors within the final latents matrices that receive the
    FFT during alpha blending.
    :param time_series_audio_vectors: See docs in the protocol.
    :param vector_length: See docs in the protocol.
    :param model_indices: See docs in the protocol.
    :return: The vector sources to be passed to the visualization functions.
    :return:
    """

    final_latents = projection_file_reader.load_final_latents_matrices_label(
        projection_file_path=projection_file_path
    )

    spectrogram = _create_spectrogram(
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        fft_amplitude_range=fft_amplitude_range,
        fft_roll_enabled=fft_roll_enabled,
    )

    num_vectors = int(spectrogram.shape[0] / vector_length)

    projected_vectors: ConcatenatedMatrices = vector_sources_common.promote_to_matrix_duplicate(
        data=vector_sources_common.duplicate_to_vector_count(
            # TODO: This is a shortcut that we can take because we know the vectors within the
            # matrix are identical.
            data=vector_sources_common.demote_to_vector_select(final_latents.data, index_to_take=0),
            vector_length=vector_length,
            target_vector_count=num_vectors,
        ).vectors,
        target_depth=final_latents.data.shape[0],
    )

    alpha_blended = vector_sources_common.promote_to_matrix_duplicate(
        ConcatenatedVectors(
            vector_sources_common.demote_to_vector_select(projected_vectors, 0) * (1.0 - alpha)
            + spectrogram * alpha
        ),
        blend_depth,
    )

    combined = np.concatenate(
        (alpha_blended, projected_vectors[blend_depth:18])  # pylint: disable=unsubscriptable-object
    )

    indices_layers: ResultLayers = model_index_selector(
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        reducer=reduce_vector_rms_rolling_average,
        model_indices=model_indices,
    )

    return VisualizationInput(
        a_vectors=VectorsLabel(
            data=spectrogram, vector_length=vector_length, label="Rolled Audio Spectrogram"
        ),
        b_vectors=MatricesLabel(
            data=projected_vectors,
            vector_length=vector_length,
            label=final_latents.label,
        ),
        combined=MatricesLabel(
            data=combined,
            vector_length=vector_length,
            label=f"Combined w/ Alpha Blending, a={alpha}",
        ),
        model_indices=indices_layers.result,
        model_index_layers=indices_layers.layers,
    )
