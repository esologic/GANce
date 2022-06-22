"""
Functions that produce `VisualizationInput`s, which are the input to module visualizations.
These are basically transformations of time series audio vectors into things like FFTs etc.
Functions themselves match a standard format so they can be used interchangeably.
"""

from typing import Any, List, Tuple, cast

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import minmax_scale
from typing_extensions import Protocol

from gance.apply_spectrogram import compute_spectrogram_smooth_scale
from gance.data_into_network_visualization.visualization_common import (
    ResultLayers,
    VisualizationInput,
)
from gance.vector_sources import vector_sources_common
from gance.vector_sources.primatives import Sigmas, gaussian_data
from gance.vector_sources.vector_reduction import (
    quantize_results_layers,
    reduce_vector_rms_rolling_average,
)
from gance.vector_sources.vector_types import (
    ConcatenatedMatrices,
    ConcatenatedVectors,
    MatricesLabel,
    SingleVector,
    VectorsLabel,
)


class CreateVisualizationInput(Protocol):  # pylint: disable=too-few-public-methods
    """
    Defines the standard shape of a visualization input function.
    """

    def __call__(
        self: "CreateVisualizationInput",
        time_series_audio_vectors: npt.NDArray[Any],
        vector_length: int,
        network_indices: List[int],
    ) -> VisualizationInput:
        """
        :param time_series_audio_vectors: The input audio file in time series form. Shouldn't
        be a spectrogram etc.
        :param vector_length: The length of the input vector to the network.
        :param network_indices: The indices of the candidate networks to be chosen from to render
        an image.
        :return: A NamedTuple for holding the result, each part is consumed in a different way.
        """


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
        roll_values: ResultLayers = quantize_results_layers(
            results_layers=reduce_vector_rms_rolling_average(
                time_series_audio_vectors=time_series_audio_vectors, vector_length=vector_length
            ),
            network_indices=list(np.arange(0, 3)),
        )

        spectrogram = vector_sources_common.smooth_each_vector(
            data=vector_sources_common.rotate_vectors_over_time(
                data=spectrogram,
                vector_length=vector_length,
                roll_values=cast(SingleVector, roll_values.result.data),
            ),
            vector_length=vector_length,
        )

    return spectrogram


def alpha_blend_vectors_max_rms_power_audio(
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    time_series_audio_vectors: ConcatenatedVectors,
    vector_length: int,
    network_indices: List[int],
) -> VisualizationInput:
    """
    For vectors:
        * Converts audio to spectrogram.
        * Applies smoothing functions to spectrogram.
        * Creates a gaussian smoothed noise source.
        * Uses alpha blending to combine these two vector sources.

    For index:
        * A rolling max rms power is computed on audio
        * This value is scaled to the range of network indices and quantized to actually select
        the networks.

    :param alpha: 0 means no music will be visible in combined signal, 1 means combined signal
    will be entirely music.
    :param fft_roll_enabled: If true, the FFT vectors move over time.
    :param fft_amplitude_range: Values in FFT are scaled to this range.
    :param fft_depth: Number of vectors within the final latents matrices that receive the
    FFT during alpha blending.
    :param time_series_audio_vectors: See docs in the protocol.
    :param vector_length: See docs in the protocol.
    :param network_indices: See docs in the protocol.
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

    indices_layers: ResultLayers = quantize_results_layers(
        results_layers=reduce_vector_rms_rolling_average(
            time_series_audio_vectors=time_series_audio_vectors, vector_length=vector_length
        ),
        network_indices=network_indices,
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
        network_indices=indices_layers,
    )


def alpha_blend_projection_file(  # pylint: disable=too-many-locals
    final_latents_matrices_label: MatricesLabel,
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    blend_depth: int,
    time_series_audio_vectors: ConcatenatedVectors,  # required by protocol
    vector_length: int,  # required by protocol
    network_indices: List[int],  # required by protocol
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
        * This value is scaled to the range of network indices and quantized to actually select
        the networks.

    :param final_latents_matrices_label: Concatenated final latents from a projection file.
    :param alpha: 0 means no music will be visible in combined signal, 1 means combined signal
    will be entirely music.
    :param fft_roll_enabled: If true, the FFT vectors move over time.
    :param fft_amplitude_range: Values in FFT are scaled to this range.
    :param blend_depth: Number of vectors within the final latents matrices that receive the
    FFT during alpha blending.
    :param time_series_audio_vectors: See docs in the protocol.
    :param vector_length: See docs in the protocol.
    :param network_indices: See docs in the protocol.
    :return: The vector sources to be passed to the visualization functions.
    """

    spectrogram = _create_spectrogram(
        time_series_audio_vectors=time_series_audio_vectors,
        vector_length=vector_length,
        fft_amplitude_range=fft_amplitude_range,
        fft_roll_enabled=fft_roll_enabled,
    )

    num_vectors = int(vector_sources_common.underlying_length(spectrogram) / vector_length)

    projected_vectors: ConcatenatedMatrices = vector_sources_common.promote_to_matrix_duplicate(
        data=vector_sources_common.duplicate_to_vector_count(
            # Note: This is a shortcut that we can take because we know the vectors within the
            # matrix are identical.
            data=vector_sources_common.demote_to_vector_select(
                final_latents_matrices_label.data, index_to_take=0
            ),
            vector_length=vector_length,
            target_vector_count=num_vectors,
        ),
        target_depth=final_latents_matrices_label.data.shape[0],
    )

    alpha_blended = vector_sources_common.promote_to_matrix_duplicate(
        ConcatenatedVectors(
            vector_sources_common.demote_to_vector_select(projected_vectors, 0) * (1.0 - alpha)
            + spectrogram * alpha
        ),
        blend_depth,
    )

    combined = np.concatenate(  # type: ignore[no-untyped-call]
        (alpha_blended, projected_vectors[blend_depth:18])  # pylint: disable=unsubscriptable-object
    )

    indices_layers: ResultLayers = quantize_results_layers(
        results_layers=reduce_vector_rms_rolling_average(
            time_series_audio_vectors=time_series_audio_vectors,
            vector_length=vector_length,
            savgol_window_length=3,
            savgol_polyorder=2,
        ),
        network_indices=network_indices,
    )

    return VisualizationInput(
        a_vectors=VectorsLabel(
            data=spectrogram, vector_length=vector_length, label="Rolled Audio Spectrogram"
        ),
        b_vectors=MatricesLabel(
            data=projected_vectors,
            vector_length=vector_length,
            label=final_latents_matrices_label.label,
        ),
        combined=MatricesLabel(
            data=combined,
            vector_length=vector_length,
            label=f"Combined w/ Alpha Blending, a={alpha}",
        ),
        network_indices=indices_layers,
    )
