"""
Common types, constants, functions used in visualization, to avoid cyclic imports.
"""

from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
import PIL
from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing_extensions import Protocol

from gance.gance_types import RGBInt8ImageType
from gance.vector_sources.vector_types import (
    MatricesLabel,
    SingleMatrix,
    SingleVector,
    VectorsLabel,
)

STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE = 10
STANDARD_MATPLOTLIB_DPI = 100


class DataLabel(NamedTuple):
    """
    Maps a piece of data to a label string.
    Label is only consumed by visualizations.
    """

    data: Union[np.ndarray, SingleVector, SingleMatrix]
    label: str


class ResultLayers(NamedTuple):
    """
    An intermediate type, linking a result of an operation with the intermediate parts.
    The `layers` field should only be re-consumed in visualization.
    """

    result: DataLabel
    layers: List[DataLabel]


class VectorsReducer(Protocol):  # pylint: disable=too-few-public-methods
    """
    Commonly defines the shape of functions to reduce pieces of audio into a single value.
    """

    def __call__(self, time_series_audio_vectors: np.ndarray, vector_length: int) -> ResultLayers:
        """
        Given an array of time series audio vectors delineated by vector_length,
        return a list of values that are representative of each subsequent vector in the input.
        The first value in the output maps to the first vector in the input etc.
        :param time_series_audio_vectors: The audio file as time series (so not an fft) vectors.
        :return: The reduction result for the given piece of audio.
        """


class VisualizationInput(NamedTuple):
    """
    Used to help visualize the combination of two sets of vectors. The `a_vectors` and `b_vectors`
    are combined to created `combined`. The function to do the combination is upstream.
    """

    # The left side of the combination, not actually fed into the rendering.
    a_vectors: Union[VectorsLabel, MatricesLabel]

    # The right side of the combination, not actually fed into the rendering.
    b_vectors: Union[VectorsLabel, MatricesLabel]

    # The result of the combination, vectors from this array are picked off and fed into the model
    # to create images.
    combined: Union[VectorsLabel, MatricesLabel]

    # Should be one integer per vector, this contains which model should be used per frame.
    model_indices: DataLabel

    # Consumed only by data visualization, not actually fed into the synthesis.
    # A list of `DataLabel` NTs that represent the different transformations on the input
    # vectors that led to the output values stored in `model_index`.
    model_index_layers: List[DataLabel]


class FrameInput(NamedTuple):
    """
    Represents all of the data needed to render a single frame of visualization.
    A `VisualizationInput` NT is split up into these, and then fed into the visualization pipeline.
    """

    # The frame number in the output video this data represents.
    frame_index: int

    # The piece of `a_vectors` that will be used in this frame.
    a_sample: DataLabel

    # The piece of `b_vectors` that will be used in this frame.
    b_sample: DataLabel

    # The piece of `combined` that will be used in this frame.
    combined_sample: DataLabel

    # Which model should be used to synthesize the resulting image.
    model_index: int

    # Holds a number of `model_index` values before and after `frame_index`.
    # Consumed by data visualization to show the history of the model index for any given frame.
    surrounding_model_indices: np.ndarray

    # Exactly like in `VisualizationInput`, but in the same range of indices as
    # `surrounding_model_indices`, so you can see the same context.
    model_index_layers: List[DataLabel]


class ConfiguredAxes(NamedTuple):
    """
    Intermediate type, stores configured axes to be then fed visualization data.
    """

    axis_3d: Optional[Axes] = None
    a_2d_axis: Optional[Axes] = None
    b_2d_axis: Optional[Axes] = None
    combined_2d_axis: Optional[Axes] = None
    model_index_plot_axis: Optional[Axes] = None
    current_model_index_plot_axis: Optional[Axes] = None
    model_selection_context: Optional[Axes] = None


def render_current_matplotlib_frame(fig: Figure, resolution: Tuple[int, int]) -> RGBInt8ImageType:
    """
    Renders the current state of the matplotlib visualization in `fig` to an image, for writing
    to a video or disk etc.
    :param fig: Figure to get an image of.
    :param resolution: Image is scaled to this resolution before being returned.
    :return: The image as a numpy array. Read the code to get to the format, kind of complicated.
    """

    # This saves the image of the graphs into memory, so it's okay to modify the plots
    # after this point.
    fig.canvas.draw()

    return RGBInt8ImageType(
        cv2.resize(
            np.array(
                PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            ),
            resolution,
        )
    )


def standard_matplotlib_figure() -> Figure:
    """
    Standard square aspect ratio matplotlib figure used in a few places in this repo.
    :return: The configured figure.
    """

    return plt.figure(
        figsize=(STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE, STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE),
        dpi=STANDARD_MATPLOTLIB_DPI,
        constrained_layout=False,  # Lets us use `.tight_layout()` later.
    )
