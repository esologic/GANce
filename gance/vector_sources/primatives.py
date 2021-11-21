"""
Functions to generate vector arrays of primative shapes. Lines, sweeps, noise etc.
"""

from typing import NamedTuple, Optional

import numpy as np
import scipy

from gance.data_into_model_visualization.vectors_to_image import (
    multi_plot_vectors,
    visualize_data_with_spectrogram_and_3d_vectors,
)
from gance.vector_sources.vector_sources_common import sub_vectors
from gance.vector_sources.vector_types import ConcatenatedVectors, SingleVector, VectorsLabel

DEFAULT_RANDOM_SEED = 1234


def line_sweep(
    start_value: int, stop_value: int, vector_length: int, num_vectors: int
) -> ConcatenatedVectors:
    """
    Create a horizontal straight line in Z that as y increases moves between two values in Z
    :param start_value: The z value at the start of the sweep.
    :param stop_value: The z value at the end of the sweep.
    :param vector_length: The length of the vector in x
    :param num_vectors: The num y points to sweep over.
    :return: An array containing the sweep
    """

    return ConcatenatedVectors(
        np.repeat(np.linspace(start_value, stop_value, vector_length), num_vectors)
    )


class Sigmas(NamedTuple):
    """
    Intermediate type, links like args.
    """

    # How alike one point will be to the next in the subsequent vector.
    across_vectors: int

    # How alike one point will be to the next within the same vector.
    within_vectors: int


def gaussian_data(
    vector_length: int,
    num_vectors: int,
    sigmas: Sigmas = Sigmas(20, 0),
    random_state: Optional["np.random.RandomState"] = None,
) -> ConcatenatedVectors:
    """
    Create gaussian field data in a format roughly analagous to wav file data so it can
    be used in a similar way.
    :param vector_length: The number of samples inside a chunk.
    :param num_vectors: the number of chunks to make data for.
    :param sigmas: Controls smoothing across the random noise. See the NT docs for explanation.
    :param random_state: Seeded randomness source.
    :return: A (1, samples_per_chunk * num_chunks) array of the gaussian data.
    """

    if random_state is None:
        random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)  # pylint: disable=no-member

    all_latents = random_state.randn(num_vectors, 1, vector_length).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(
        input=all_latents, sigma=(sigmas.across_vectors, 0, sigmas.within_vectors), mode="wrap"
    )
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    return ConcatenatedVectors(all_latents.reshape(vector_length * num_vectors))


def _single_square_wave_vector(
    rising_edge_x: int, falling_edge_x: int, y_offset: float, y_amplitude: float, vector_length: int
) -> SingleVector:
    """
    Create an individual square wave vector.
    :param rising_edge_x: inclusive! Point where value transitions from `off_value` -> `on_value`.
    :param falling_edge_x: inclusive! Point where value transitions from `on_value` -> `off_value`.
    :param y_offset: The `off` value of the square wave.
    :param y_amplitude: The `on` value of the square wave.
    :param vector_length: The length of the output vector.
    :return: A single vector.
    """

    return SingleVector(
        np.array(
            [
                y_amplitude if rising_edge_x <= i <= falling_edge_x else y_offset
                for i in range(vector_length)
            ]
        )
    )


def square_wave_sweep_horizontal(
    vector_length: int, pulse_width: int, y_offset: int = 0, y_amplitude: int = 10
) -> ConcatenatedVectors:
    """
    Sweep a square wave across a vector from left to right. Can't do steps smaller than one because
    we're transversing an index space.
    :param vector_length: The length of the output vector.
    :param pulse_width: The length of the pulse.
    :param y_offset: The `off` value of the square wave.
    :param y_amplitude: The `on` value of the square wave.
    :return: An array of vectors.
    """

    return ConcatenatedVectors(
        np.concatenate(
            [
                _single_square_wave_vector(
                    rising_edge_x=value,
                    falling_edge_x=value + pulse_width,
                    y_amplitude=y_amplitude,
                    y_offset=y_offset,
                    vector_length=vector_length,
                )
                for value in np.arange(y_offset, y_amplitude)
            ]
        )
    )


def square_wave_sweep_vertical(
    vector_length: int,
    rising_edge_x: int,
    pulse_width: int,
    y_offset: int = -10,
    y_amplitude: int = 10,
    step_size: float = 1.0,
) -> ConcatenatedVectors:
    """
    Sweep a square wave from a line to the given amplitude, each vector the amplitude of the wave
    increases by `step_size`.
    :param vector_length: The length of the output vector.
    :param rising_edge_x: The start value of the wave.
    :param pulse_width: The length of the pulse.
    :param y_offset: The `off` value of the square wave.
    :param y_amplitude: The `on` value of the square wave.
    :param step_size: The difference in amplitude between adjacent waves.
    :return: An array of vectors.
    """

    return ConcatenatedVectors(
        np.concatenate(
            [
                _single_square_wave_vector(
                    y_offset=y_offset,
                    y_amplitude=value,
                    vector_length=vector_length,
                    rising_edge_x=rising_edge_x,
                    falling_edge_x=rising_edge_x + pulse_width,
                )
                for value in np.arange(y_offset, y_amplitude, step_size)
            ]
        )
    )


def single_sine_wave_vector(vector_length: int, y_amplitude: float) -> SingleVector:
    """
    Creates a sine wave across the vector length with a given amplitude.
    Do not use this function! TODO: finish this implementation.
    :param vector_length: The length of the vector.
    :param y_amplitude: Amplitude of the resulting wave.
    :return: The vector.
    """

    x_values = np.arange(0, vector_length, 1)
    return SingleVector(np.sin(x_values) * y_amplitude)


def vertical_sweep_demo() -> None:
    """
    Demo function for `square_wave_sweep_vertical`, used for debugging.
    :return: None
    """

    vector_length = 100

    sin_vector = single_sine_wave_vector(vector_length, y_amplitude=1)

    data = np.concatenate(
        [
            [
                vector_point + sin_point if vector_point != 0 else 0
                for vector_point, sin_point in zip(vector, sin_vector)
            ]
            for vector in sub_vectors(
                data=square_wave_sweep_vertical(
                    vector_length=vector_length,
                    rising_edge_x=0,
                    pulse_width=50,
                    y_offset=0,
                    y_amplitude=5,
                    step_size=1.0,
                ),
                vector_length=vector_length,
            )
        ]
    )

    visualize_data_with_spectrogram_and_3d_vectors(
        vectors_label=VectorsLabel(
            data=ConcatenatedVectors(data), vector_length=vector_length, label="Vertical Sweep"
        ),
        inline_spectrogram=False,
    )


def sigmas_demo() -> None:
    """
    Small demo to visualize the effect the different sigma parameters have.
    :return: None
    """

    vector_length = 1024

    multi_plot_vectors(
        [
            VectorsLabel(
                data=gaussian_data(
                    vector_length=vector_length, num_vectors=200, sigmas=Sigmas(sigma, sigma)
                ),
                label=f"gaussian, sigma: {sigma}",
                vector_length=vector_length,
            )
            for sigma in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 500]
        ],
    )


if __name__ == "__main__":
    vertical_sweep_demo()
