"""
Functions related to using sequences of vectors as input to models, creating synthesized videos.
"""

import itertools
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union, cast

import numpy as np
import PIL
from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gance.data_into_model_visualization.vectors_3d import draw_y_point, plot_vectors_3d
from gance.data_into_model_visualization.vectors_to_image import vector_visualizer
from gance.data_into_model_visualization.visualization_common import (
    STANDARD_MATPLOTLIB_DPI,
    STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE,
    ConfiguredAxes,
    DataLabel,
    FrameInput,
    VisualizationInput,
    infinite_colors,
    render_current_matplotlib_frame,
)
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.image_sources.video_common import create_video_writer
from gance.iterator_on_disk import deserialize_hdf5, serialize_hdf5
from gance.logger_common import LOGGER
from gance.model_interface.model_functions import ModelInterface, MultiModel
from gance.vector_sources.vector_sources_common import (
    demote_to_vector_select,
    pad_array,
    sub_vectors,
    underlying_vector_length,
)
from gance.vector_sources.vector_types import (
    ConcatenatedVectors,
    SingleVector,
    VectorsLabel,
    is_vector,
)


def _configure_axes(  # pylint: disable=too-many-locals
    fig: Figure,
    enable_2d: bool,
    enable_3d: bool,
    visualization_input: VisualizationInput,
    vector_length: int,
) -> ConfiguredAxes:
    """
    Configure the given figure's axes, depending on the required visualizations.
    :param fig: Figure to create axes on.
    :param enable_2d: If 2d visualizations are needed.
    :param enable_3d: If 3d visualizations are needed.
    :return: an NT that contains the resulting axes.
    """

    num_rows = 10
    full_column_width = 12
    num_columns = full_column_width if (enable_3d and enable_2d) else int(full_column_width / 2)
    column_halfway = int(num_columns / 2) if enable_3d else num_columns

    axis_3d = None
    a_2d_axis = None
    b_2d_axis = None
    combined_2d_axis = None
    model_index_plot_axis = None
    current_model_index_plot_axis = None
    model_selection_context = None

    gs = fig.add_gridspec(nrows=num_rows, ncols=num_columns)

    if enable_2d:

        a_2d_axis = fig.add_subplot(gs[0:2, 0:column_halfway])
        b_2d_axis = fig.add_subplot(gs[2:4, 0:column_halfway])
        combined_2d_axis = fig.add_subplot(gs[4:6, 0:column_halfway])

        model_selection_context = fig.add_subplot(gs[6:8, 0:column_halfway])

        model_index_plot_axis = fig.add_subplot(gs[8:9, 0:column_halfway])
        current_model_index_plot_axis = fig.add_subplot(gs[9:10, 0:column_halfway])

        for axis, data_on_axis, title in [
            (a_2d_axis, visualization_input.a_vectors, "Input A"),
            (b_2d_axis, visualization_input.b_vectors, "Input B"),
            (combined_2d_axis, visualization_input.combined, "Combined Inputs"),
        ]:
            # Why can't I use a map here...
            axis.set_xlim((0, vector_length))
            axis.set_ylim((data_on_axis.data.min(), data_on_axis.data.max()))
            axis.set_title(title)
            axis.set_xlabel("Sample # In Vector")
            axis.set_ylabel("Signal Amplitude")

        model_selection_context.set_ylim(
            (
                min([layer.data.min() for layer in visualization_input.model_indices.layers]),
                max([layer.data.max() for layer in visualization_input.model_indices.layers]),
            )
        )
        model_selection_context.set_title(
            "Composition of model index selection: "
            f"{visualization_input.model_indices.result.label}"
        )

        model_index_limits = (
            visualization_input.model_indices.result.data.min(),
            visualization_input.model_indices.result.data.max(),
        )
        model_index_plot_axis.set_ylim(model_index_limits)

        current_model_index_plot_axis.set_xlim(model_index_limits)
        current_model_index_plot_axis.set_xticks(
            np.arange(visualization_input.model_indices.result.data.max())
        )
        current_model_index_plot_axis.set_xticklabels(
            current_model_index_plot_axis.get_xticks(), rotation=90
        )
        current_model_index_plot_axis.set_ylabel("Model Index")

        if enable_3d:
            axis_3d = fig.add_subplot(gs[0:num_rows, column_halfway:num_columns], projection="3d")
            plot_vectors_3d(
                axis_3d,
                vectors_label=VectorsLabel(
                    data=visualization_input.combined.data,
                    vector_length=vector_length,
                    label=visualization_input.combined.label,
                ),
            )
    else:
        if enable_3d:
            axis_3d = fig.add_subplot(gs[0:num_rows, 0:num_columns], projection="3d")

    plt.tight_layout()

    return ConfiguredAxes(
        axis_3d=axis_3d,
        a_2d_axis=a_2d_axis,
        b_2d_axis=b_2d_axis,
        combined_2d_axis=combined_2d_axis,
        model_index_plot_axis=model_index_plot_axis,
        current_model_index_plot_axis=current_model_index_plot_axis,
        model_selection_context=model_selection_context,
    )


def _frame_inputs(
    visualization_input: VisualizationInput,
    vector_length: int,
    model_index_window_width: Optional[int] = None,
) -> List[FrameInput]:
    """
    Split a `VisualizationInput` into many `FrameInputs` that each represent the data that should
    be exposed to the visualization/model in the given frame.
    :param visualization_input: Contains the entire input to the model, as well as the entire
    inputs to the data visualization.
    :param vector_length: The length of the expected output vectors. This will determine
    how many `FrameInput`s are created.
    :param model_index_window_width: For visualization of the model index, how many indicies
    should be displayed at once on the time series plot.
    :return: One `FrameInput` per possible frames in `visualization_input`.
    """

    num_points = visualization_input.model_indices.result.data.shape[0]
    model_index_window_width = (
        model_index_window_width
        if model_index_window_width is not None
        else int(np.ceil(num_points / 5))
    )
    width = model_index_window_width * int(np.ceil(num_points / model_index_window_width))

    index_windows = sub_vectors(
        ConcatenatedVectors(pad_array(visualization_input.model_indices.result.data, width)),
        model_index_window_width,
    )

    context_windows = [
        DataLabel(
            data=sub_vectors(
                ConcatenatedVectors(pad_array(layer.data, width)),
                model_index_window_width,
            ),
            label=layer.label,
        )
        for layer in visualization_input.model_indices.layers
    ]

    def create_frame_input(
        index: int,
        a_sample: DataLabel,
        b_sample: DataLabel,
        combined_sample: DataLabel,
        model_index: int,
    ) -> FrameInput:
        """
        Wrapper function to be able to re-use index compute.
        :param index: See `FrameInput` docs.
        :param a_sample: See `FrameInput` docs.
        :param b_sample: See `FrameInput` docs.
        :param combined_sample: See `FrameInput` docs.
        :param model_index: See `FrameInput` docs.
        :return: A `FrameInput` object.
        """

        window_index = int(index / model_index_window_width)

        return FrameInput(
            frame_index=index,
            a_sample=a_sample,
            b_sample=b_sample,
            combined_sample=combined_sample,
            model_index=model_index,
            surrounding_model_indices=index_windows[window_index],
            model_index_layers=[
                DataLabel(data=context_window.data[window_index], label=context_window.label)
                for context_window in context_windows
            ],
        )

    # Split each of the vector members of `visualization_input` into `vector_length` parts.
    data_parts: List[List[DataLabel]] = [
        [
            DataLabel(vector, vectors_label.label)
            for vector in sub_vectors(ConcatenatedVectors(vectors_label.data), vector_length)
        ]
        for vectors_label in (
            visualization_input.a_vectors,
            visualization_input.b_vectors,
            visualization_input.combined,
        )
    ]

    return [
        create_frame_input(index, a_sample, b_sample, combined_sample, model_index)
        for index, (a_sample, b_sample, combined_sample, model_index) in enumerate(
            zip(*data_parts, visualization_input.model_indices.result.data),
        )
    ]


def _write_data_to_axes(
    axes: ConfiguredAxes, frame_input: FrameInput, vector_length: int
) -> Iterator[
    Union[Poly3DCollection, PathCollection, PathCollection, Line2D, BarContainer, Legend]
]:
    """
    Given some data to visualize (in `frame_input`) and some axes to write the date to
    (in `axes`), do the visualizations. If a given axis is `None` within `axes`, it's corresponding
    visualization will not be created, even if the data is present in `frame_input`.
    :return: An iterator of the resulting matplotlib-related objects. It's important that for each
    of these objects, consumer calls `.remove()` on them before creating the next frame so the
    the visualizations are accurate to the currently displayed frame. If you don't do this, it will
    just write the visualizations on top of each other.
    """

    def draw_point_on_3d_axis() -> List[Poly3DCollection]:
        """
        Draws the indicator point on the 3d axes, assumes the waveform has already been
        drawn on the axes.
        :return: A list of the matplotlib elements that must be `.remove()`d before drawing
        the next frame.
        """
        return [
            draw_y_point(
                ax=axes.axis_3d,
                # This constant moves the dot over slightly to the right so it's easier to see.
                x=int(np.ceil(vector_length + (vector_length * 0.1))),
                y=frame_input.frame_index,
            )
        ]

    def draw_legend(ax: Axes) -> Legend:
        """
        Draws a legend in the standard style on some axes.
        :param ax: Axes to draw the legend on.
        :return: The reference, make sure you `.remove()` this!
        """
        return ax.legend(loc="upper right")

    def scatter_2d(
        ax: Axes, color: str, data_label: DataLabel
    ) -> List[Union[PathCollection, Legend]]:
        """
        Draw a scatter plot of the given data on the given axes.
        :param ax: The axes to write the data onto.
        :param data_label: The data to write on the axis, and the label for the line.
        :param color: The color of the scatter plot line, matplotlib colors.
        :return: A list of the matplotlib elements that must be `.remove()`d before drawing
        the next frame.
        """

        data_is_vector = is_vector(data_label.data)
        if not data_is_vector:
            LOGGER.debug("Plot is a lie! Plotting a matrix as a vector!")

        y_values = data_label.data if data_is_vector else demote_to_vector_select(data_label.data)

        return [
            ax.scatter(
                np.arange(underlying_vector_length(data_label.data)),
                y_values,
                color=color,
                label=data_label.label,
            ),
            draw_legend(ax),
        ]

    def model_index_time_series() -> List[Union[PathCollection, Line2D]]:
        """
        Draws a window of indices, gives some context as to what is going to happen before/after
        the current frame, uses a line to show where the current frame is within the window.
        :return: A list of the matplotlib elements that must be `.remove()`d before drawing
        the next frame.
        """

        width = len(frame_input.surrounding_model_indices)

        # Draws a window of indices, gives some context as to what is going to happen before/after
        # the current frame.
        model_index_line = axes.model_index_plot_axis.scatter(
            np.arange(width), frame_input.surrounding_model_indices, color="c"
        )

        # Shows where the current frame is within the window.
        label_line = axes.model_index_plot_axis.axvline(
            x=frame_input.frame_index % width, color="r"
        )

        return [model_index_line, label_line]

    def current_model_index_indicator() -> List[BarContainer]:
        """
        Draw a horizontal bar chart to show which model index is generating the current frame.
        :return: A list of the matplotlib elements that must be `.remove()`d before drawing
        the next frame.
        """
        return [
            axes.current_model_index_plot_axis.barh(
                [0], [frame_input.model_index], color="m", align="center", tick_label=[""]
            )
        ]

    def index_context() -> List[Union[PathCollection, Line2D, Legend]]:
        """
        Draws each of the signals that make up the final index over each each other.
        :return: A list of the matplotlib elements that must be `.remove()`d before drawing
        the next frame.
        """

        width = len(frame_input.surrounding_model_indices)
        x_values = np.arange(width)

        # Draws a window of indices, gives some context as to what is going to happen before/after
        # the current frame.
        context_lines = [
            axes.model_selection_context.plot(
                x_values, layer.data, alpha=0.5, label=layer.label, color=str(color)
            )[0]
            for layer, color in zip(frame_input.model_index_layers, infinite_colors())
        ]

        return context_lines + [
            # Shows where the current frame is within the window.
            axes.model_selection_context.axvline(x=frame_input.frame_index % width, color="r"),
            draw_legend(axes.model_selection_context),
        ]

    return itertools.chain.from_iterable(
        [
            draw_function()
            for ax, draw_function in zip(
                axes,  # Axes that shouldn't have data written to them will be `None` in this NT.
                # A list of functions, that when executed, will write data to their given axis.
                # This list should match the order in `axes`.
                [
                    draw_point_on_3d_axis,
                    lambda: scatter_2d(axes.a_2d_axis, "r", frame_input.a_sample),
                    lambda: scatter_2d(axes.b_2d_axis, "g", frame_input.b_sample),
                    lambda: scatter_2d(axes.combined_2d_axis, "b", frame_input.combined_sample),
                    model_index_time_series,
                    current_model_index_indicator,
                    index_context,
                ],
            )
            if ax is not None
        ]
    )


class SynthesisOutput(NamedTuple):
    """
    Describes the two image sources that can result from a synthesis run.
    """

    synthesized_images: Optional[ImageSourceType]
    visualization_images: Optional[ImageSourceType]


class _RenderedFrame(NamedTuple):
    """
    Intermediate Type, unpacks the results from a synthesis run into their component parts.
    """

    synthesized_frame: Optional[RGBInt8ImageType]
    visualization_frame: Optional[RGBInt8ImageType]


class _FrameInputPath(NamedTuple):
    """
    Intermediate type, links a frame input to where the resulting image will be
    stored on disk.
    """

    frame: FrameInput
    path: Path


def compute_force_synthesis_order(
    force_optimize_synthesis_order: bool, num_model_indices: Optional[int]
) -> bool:
    """
    Read user input, and the state of the models to determine if synthesis ordering
    should be used.
    :param force_optimize_synthesis_order: From user.
    :param num_model_indices: From model discovery process.
    :return: The flag.
    """

    return (
        force_optimize_synthesis_order
        if num_model_indices is not None and num_model_indices > 1
        else False
    )


def load_model_image_and_delete(frame_input_path: _FrameInputPath) -> RGBInt8ImageType:
    """
    Load the pickled frame from disk, and then delete the pickle file.
    :param frame_input_path: Represents the image and the path to the rendered image on disk.
    :return: The rendered frame.
    """

    LOGGER.info(f"Loading frame from disk: #{frame_input_path.frame.frame_index}")
    output: RGBInt8ImageType = deserialize_hdf5(path=frame_input_path.path)
    frame_input_path.path.unlink()
    return output


def vector_synthesis(  # pylint: disable=too-many-locals # <------- pain
    data: VisualizationInput,
    models: Optional[MultiModel],
    default_vector_length: Optional[int] = 1024,
    video_height: Optional[int] = 1024,
    enable_3d: bool = False,
    enable_2d: bool = True,
    frames_to_visualize: Optional[int] = None,
    model_index_window_width: Optional[int] = None,
    force_optimize_synthesis_order: bool = True,
) -> SynthesisOutput:
    """
    Given an input array, for each possible input vector in the array, feed these vectors into
    the model. Take the output image and combine it with a matplotlib visualization of the entire
    data array in 3d, as well as the input vector plotted in 2d. Combine these images into a
    video.

    If no model is given, creates a video out of only the matplotlib components.

    Since model switching is a costly operation, frames are first written to disk as pickled objects
    sorted by their model index, so model switching only needs to happen once per the number of
    input models. The images are then loaded back into memory and written to the output video file
    in the order they should be displayed. The intermediate pickled objects are all created in
    a `TemporaryDirectory` and are destroyed after they fall out of scope.

    Note: This is the most complicated function in the whole project.

    :param data: The data array to visualize.
    :param models: The face-generating model.
    :param default_vector_length: If no model is given, this will be used as the length
    of the "input vector"s.
    :param video_height: The individual matplotlib plots, as well as the output from the model
    are both squares with a side length of this many pixels.
    :param video_fps: The frames per second of the output video.
    :param enable_3d: If True, a 3d visualization of the input vectors will be created alongside
    the output of the model (if the `models` is not None).
    :param enable_2d: If True, a 2d visualization of the combination of the input vectors
    will be created alongside the output of the model.
    :param frames_to_visualize: The number of frames in the input to visualize. Starts from the
    first index, goes to this value. Ex. 10 is given, the first 10 frames will be visualized.
    :param model_index_window_width: For the time-series visualization of the model index, this is
    how many indices should be displayed at once.
    :param force_optimize_synthesis_order: If the snythesis order optimization should
    be done or not.
    :return: `output_video_path`, but now the video will actually be there.
    """

    if not enable_3d and not enable_2d and models is None:
        raise ValueError("Nothing to render!")

    vector_length = default_vector_length if models is None else models.expected_vector_length

    total_num_frames = (
        len(sub_vectors(data=data.combined.data, vector_length=vector_length))
        if frames_to_visualize is None
        else frames_to_visualize
    )

    if frames_to_visualize is not None:
        LOGGER.warning(f"Truncating output to the first {frames_to_visualize} frames!")

    input_sources = iter(
        itertools.tee(
            itertools.islice(
                _frame_inputs(
                    visualization_input=data,
                    vector_length=vector_length,
                    model_index_window_width=model_index_window_width,
                ),
                frames_to_visualize,
            ),
            sum([enable_3d or enable_2d, models is not None]),
        )
    )

    def create_visualization_frames(
        frame_inputs: Iterator[FrameInput],
    ) -> ImageSourceType:
        """
        Visualize the data that will be fed into the synthesis process.
        :param frame_inputs: To visualize.
        :return: Visualization images.
        """

        # Needs to be this aspect ratio
        fig = plt.figure(
            figsize=(
                STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE * 2
                if (enable_3d and enable_2d)
                else STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE,
                STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE,
            ),
            dpi=STANDARD_MATPLOTLIB_DPI,
            constrained_layout=False,  # Lets us use `.tight_layout()` later.
        )

        configured_axes = _configure_axes(
            fig=fig,
            enable_2d=enable_2d,
            enable_3d=enable_3d,
            visualization_input=data,
            vector_length=vector_length,
        )

        data_visualizations_resolution = (
            video_height * sum([1 if enable_2d else 0, 1 if enable_3d else 0]),
            video_height,
        )

        for index, frame_input in enumerate(frame_inputs):

            drawn_elements = _write_data_to_axes(
                axes=configured_axes,
                frame_input=frame_input,
                vector_length=vector_length,
            )

            LOGGER.info(f"Visualizing synthesis input #{index}")

            yield render_current_matplotlib_frame(
                fig=fig, resolution=data_visualizations_resolution
            )

            # Remove these plots so they're not visible on the next frame.
            for element in drawn_elements:
                element.remove()

    def create_model_frames(frame_inputs: Iterator[FrameInput]) -> ImageSourceType:
        """
        Create an iterator what when consumed yields synthesized frames from the model
        given the input data per frame.
        :param frame_inputs: To feed into the model.
        :return: Iterator of images from the model.
        """

        def render_model_frame_in_memory(frame_input: FrameInput) -> RGBInt8ImageType:
            """
            Render the given `FrameInput` to an actual image given the context of the run.
            This creates the matplotlib visualizations (if requested) and synthesizes the image out
            of the model (if requested).
            :param frame_input: The information needed to create the frame.
            :return: The resulting image (frame) as a numpy array.
            """

            LOGGER.info(f"Using model to synthesize frame #{frame_input.frame_index}")

            if underlying_vector_length(frame_input.combined_sample.data) != vector_length:
                LOGGER.warning(
                    f"Bad Sample Shape, expected {vector_length}, "
                    f"got {frame_input.combined_sample.data.shape}"
                )

            return cast(
                RGBInt8ImageType,
                cv2.resize(
                    models.indexed_create_image_generic(  # This is the call that uses the model.
                        index=frame_input.model_index,
                        data=frame_input.combined_sample.data,
                    ),
                    (video_height, video_height),
                ),
            )

        def serialize_frame(frame_input: FrameInput, rendered_frame_count: int) -> _FrameInputPath:
            """
            Renders a given frame into memory, and immediately writes it to disk as a pickled file.
            :param frame_input: The frame to render.
            :param rendered_frame_count: Consumed in logging.
            :return: A tuple, the `FrameInput` and a path to the pickle file on disk.
            """

            LOGGER.info(
                "Serializing frame to file. "
                f"Model index: {frame_input.model_index}. "
                f"Frame position: {frame_input.frame_index}. "
                f"Frame count: {rendered_frame_count}/{total_num_frames}. "
            )

            # These will get deleted later in the pipeline.
            with NamedTemporaryFile(mode="w", delete=False) as p:
                frame_path = Path(p.name)
                frame = render_model_frame_in_memory(frame_input)
                serialize_hdf5(path=frame_path, item=frame)
                return _FrameInputPath(frame=frame_input, path=frame_path)

        if compute_force_synthesis_order(
            force_optimize_synthesis_order,
            len(models.model_indices) if models is not None else None,
        ):
            LOGGER.info("Will synthesis order will be optimized")

            efficient_rendering_order = (
                sorted(frame_inputs, key=lambda frame_input: frame_input.model_index)
                if models is not None
                else frame_inputs
            )

            files_on_disk: Iterator[_FrameInputPath] = (
                serialize_frame(frame_input, index)
                for index, frame_input in enumerate(efficient_rendering_order)
            )

            # The `sorted` operation here is what causes the frames to render.
            return map(
                load_model_image_and_delete,
                sorted(files_on_disk, key=lambda frame_path: frame_path[0].frame_index),
            )
        else:
            LOGGER.info("Will synthesis order will not be optimized")
            return map(render_model_frame_in_memory, frame_inputs)

    return SynthesisOutput(
        synthesized_images=create_model_frames(next(input_sources)) if models is not None else None,
        visualization_images=create_visualization_frames(next(input_sources))
        if (enable_2d or enable_3d)
        else None,
    )


def _y_bounds(data: np.ndarray, y_range: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Process UI input into the set of Y bounds that should be used.
    :param data: Will be considered if no range is given.
    :param y_range: Explicit y range
    :return: Range to be used
    """

    return y_range if y_range is not None else data.min(), data.max()


def vectors_single_model_visualization(  # pylint: disable=too-many-locals
    vectors_label: VectorsLabel,
    output_video_path: Path,
    model: ModelInterface,
    video_height: Optional[int] = 1024,
    y_range: Optional[Tuple[int, int]] = None,
    video_fps: float = 60.0,
) -> None:
    """
    Creates a very simple video:
     * The left side is a scatter plot of the current vector
     * The right side is the output from the model given the current vector
    Allows user to quickly understand the effect a set of vectors have on a model.
    Note: Can't use a matrix input here per the types.
    :param vectors_label: The vectors the visualize/use for synthesis.
    :param output_video_path: The path to write the resulting video to.
    :param model: The model to use for synthesis.
    :param video_height: The height of the output video in pixels, the width will be 2x the height.
    :param y_range: If given, the y range will be clamped to this min max pair.
    :param video_fps: FPS of output video.
    :return: None
    """

    y_min, y_max = _y_bounds(data=vectors_label.data, y_range=y_range)

    make_visualization = vector_visualizer(
        y_min=y_min,
        y_max=y_max,
        title=vectors_label.label,
        output_width=video_height,
        output_height=video_height,
    )

    video = create_video_writer(
        video_path=output_video_path,
        num_squares_width=2,
        video_fps=video_fps,
        video_height=video_height,
    )

    all_vectors = sub_vectors(data=vectors_label.data, vector_length=model.expected_vector_length)

    num_vectors = len(all_vectors)
    x_values = np.arange(vectors_label.vector_length)

    for index, vector in enumerate(all_vectors):
        logging.info(f"Writing video: {output_video_path.name}, frame: {index}/{num_vectors}")

        with make_visualization(x_values=x_values, y_values=vector) as visualization:
            # Puts the data visualization to the left of the synthesis.
            frame = cv2.hconcat([visualization, model.create_image_vector(vector)])

        video.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

    video.release()


def single_vector_single_model_visualization(
    vector: SingleVector,
    title: str,
    output_image_path: Path,
    model: ModelInterface,
    image_height: Optional[int] = 1024,
    y_range: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Creates a very simple image:
     * The left side is a scatter plot of the vector.
     * The right side is the output from the model given the current vector
    Allows user to quickly understand the effect a set of vectors have on a model.
    :param vector: The data to plot.
    :param title: The title to put on the resulting plot.
    :param output_image_path: The path to write the resulting image to.
    :param model: The model to use for synthesis.
    :param image_height: The height of the output video in pixels, the width will be 2x the height.
    :param y_range: If given, the y range will be clamped to this min max pair.
    :return: None
    """

    y_min, y_max = _y_bounds(data=vector, y_range=y_range)

    make_visualization = vector_visualizer(
        y_min=y_min,
        y_max=y_max,
        title=title,
        output_width=image_height,
        output_height=image_height,
    )

    x_values = np.arange(vector.shape[-1])

    with make_visualization(x_values=x_values, y_values=vector) as visualization:
        # Puts the data visualization to the left of the synthesis.
        frame = cv2.hconcat([visualization, model.create_image_generic(vector)])

    i = PIL.Image.fromarray(frame.astype(np.uint8))
    i.save(str(output_image_path))
