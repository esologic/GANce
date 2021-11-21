"""
WIP module for looking at projection files and doing some analysis.
"""

import itertools
import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from cv2 import cv2
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from gance.data_into_model_visualization.vectors_to_image import SingleVectorViz, vector_visualizer
from gance.data_into_model_visualization.visualization_common import (
    STANDARD_MATPLOTLIB_DPI,
    STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE,
)
from gance.hash_file import hash_file
from gance.logger_common import LOGGER
from gance.model_interface.model_functions import (
    ModelInterfaceInProcess,
    create_model_interface_process,
)
from gance.projection.projection_file_reader import (
    ProjectionFileReader,
    final_latents_matrices_label,
    load_projection_file,
)
from gance.vector_sources.vector_sources_common import sub_vectors
from gance.vector_sources.vector_types import MatricesLabel, SingleMatrix
from gance.video_common import create_video_writer


def _spline_to_points(splines: List[UnivariateSpline], x_values: np.ndarray) -> List[List[float]]:
    """
    Helper function to enumerate the values in a list of splines.
    :param splines: Splines to process.
    :return: List of values for the splines.
    """
    return [spline(x_values) for spline in splines]


class _VerticalLinesDescription(NamedTuple):
    """
    Intermediate type, describes a group of vertical lines for a matplotlib visualization.
    """

    x_positions: Union[List[float], List[int]]
    color: str
    line_styles: str
    width: float
    alpha: float
    label: str


def visualize_projection_convergence(  # pylint: disable=too-many-locals
    projection_file_path: Path,
    output_image_path: Optional[Path] = None,
    consider_first_n_frames: Optional[int] = None,
) -> None:
    """
    Shows each projection history latent over time as it approaches the target.
    :param projection_file_path: Path to file to read.
    :param output_image_path: The path to write the resulting visualization image to, if not
    given the matplotlib preview window will be opened.
    :param consider_first_n_frames: The first n amount of frames will be visualized in this
    analysis. If not given all frames will be considered.
    :return: None
    """

    with load_projection_file(projection_file_path) as reader:

        raw_data_lines = [
            np.array([np.sum(abs((final_latents - latent))) for latent in latent_history])
            for latent_history, final_latents in itertools.islice(
                zip(reader.latents_histories, reader.final_latents), consider_first_n_frames
            )
        ]

        if not raw_data_lines or not reader.projection_attributes.latents_histories_enabled:
            raise ValueError("File doesn't contain the data to visualize.")

    # We can safely do this index because of the previous check
    x_values = np.arange(len(raw_data_lines[0]))

    first_order_derivatives, second_order_derivatives = [
        [
            UnivariateSpline(
                x=x_values, y=line, s=5 if len(line) >= 5 else len(line) - 1
            ).derivative(n=derivative)
            for line in raw_data_lines
        ]
        for derivative in (1, 2)
    ]

    # Points where the projection is 80% complete. This isn't exact, but it's a heuristic
    # that tracked with visual results.
    points_of_interest = [
        np.where(line <= (line.max() - line.min()) * 0.2)[0][0] for line in raw_data_lines
    ]

    average = int(np.mean(points_of_interest))
    standard_deviation = int(np.std(points_of_interest))

    # Needs to be this aspect ratio, would be easy to pass these in if needed later on.
    fig = plt.figure(
        figsize=(STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE, STANDARD_MATPLOTLIB_SIDE_LENGTH_FIGSIZE),
        dpi=STANDARD_MATPLOTLIB_DPI,
        constrained_layout=False,  # Lets us use `.tight_layout()` later.
    )

    fig.suptitle(
        os.linesep.join(
            [
                f"File: {projection_file_path.name}",
                f"Average 80% Projection Step: {average}",
            ]
        )
    )

    one_std_deviation_below_mean = average - standard_deviation
    two_std_deviation_below_mean = average - standard_deviation * 2

    lines_descriptions = [
        _VerticalLinesDescription(
            x_positions=points_of_interest,
            color="grey",
            line_styles="dotted",
            width=1.0,
            alpha=0.5,
            label="Frame 80% projected",
        ),
        _VerticalLinesDescription(
            x_positions=[average],
            color="black",
            line_styles="solid",
            width=2.0,
            alpha=1,
            label=f"Average ({average})",
        ),
        _VerticalLinesDescription(
            x_positions=[one_std_deviation_below_mean],
            color="blue",
            line_styles="solid",
            width=2.0,
            alpha=1,
            label=f"1 Std. Deviation Below Mean ({one_std_deviation_below_mean})",
        ),
        _VerticalLinesDescription(
            x_positions=[two_std_deviation_below_mean],
            color="Purple",
            line_styles="solid",
            width=2.0,
            alpha=1,
            label=f"2 Std. Deviation Below Mean ({two_std_deviation_below_mean})",
        ),
    ]

    for axis, title, lines, compute_limits in zip(
        fig.subplots(nrows=3, ncols=1, sharex=True),
        (
            "Total Numerical Difference Between Projected Latents and Final Latents Per Frame",
            "First Order Derivation (Slope)",
            "Second Order Derivation (Curvature)",
        ),
        (
            raw_data_lines,
            _spline_to_points(first_order_derivatives, x_values),
            _spline_to_points(second_order_derivatives, x_values),
        ),
        (False, True, True),
    ):

        all_points = np.concatenate(lines)

        if compute_limits:
            mean = all_points.mean()
            bound = all_points.std() * 5
            plot_min = mean - bound
            plot_max = mean + bound
            axis.set_ylim(bottom=plot_min, top=plot_max)
        else:
            plot_min = all_points.min()
            plot_max = all_points.max()

        axis.set_xlabel("Projection Step")
        axis.set_title(title)

        # Might want to label the lines here which is why I'm keeping the `frame_index` around.
        for line, frame_color in zip(lines, itertools.cycle(list(mcolors.XKCD_COLORS.keys()))):
            axis.plot(line, color=frame_color, alpha=0.5)

        for lines_description in lines_descriptions:
            axis.vlines(
                x=lines_description.x_positions,
                ymin=plot_min,
                ymax=plot_max,
                linestyles=lines_description.line_styles,
                color=lines_description.color,
                alpha=lines_description.alpha,
                label=lines_description.label,
                linewidth=lines_description.width,
            )

        axis.grid()

    plt.legend()

    if output_image_path:
        plt.savefig(fname=str(output_image_path))
    else:
        plt.show()


def visualize_final_latents(
    projection_file_path: Path,
    output_video_path: Path,
    video_height: Optional[int] = 1024,
) -> None:
    """
    Visualize a projection file. Makes a video comparing the target with the final projection,
    also displays the final latents.
    :param projection_file_path: Path to the projection file.
    :param output_video_path: The path to write the resulting video to.
    :param video_height: The height of the output video in pixels, the width will be 2x the height.
    :return: None
    """

    with load_projection_file(projection_file_path) as reader:

        matrices_label = final_latents_matrices_label(reader)

        make_visualization = vector_visualizer(
            y_min=matrices_label.data.min(),
            y_max=matrices_label.data.max(),
            title=matrices_label.label,
            output_width=video_height,
            output_height=video_height,
        )

        video = create_video_writer(
            video_path=output_video_path,
            num_squares=3,
            video_fps=reader.projection_attributes.projection_fps,
            video_height=video_height,
        )

        all_matrices = sub_vectors(
            data=matrices_label.data, vector_length=matrices_label.vector_length
        )

        num_matrices = len(all_matrices)

        for index, (latents, target, final_image) in enumerate(
            zip(all_matrices, reader.target_images, reader.final_images)
        ):

            with make_visualization(
                x_values=np.arange(matrices_label.vector_length), y_values=SingleMatrix(latents)
            ) as visualization:
                # Puts the data visualization to the left of the synthesis.
                frame = cv2.hconcat([visualization, target, final_image])

            video.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

            LOGGER.info(f"Wrote frame: {output_video_path}, frame: {index + 1}/{num_matrices}")

        video.release()


def _setup_visualization(
    model_not_matching_ok: bool,
    projection_model_path: Path,
    projection_file_reader: ProjectionFileReader,
    video_height: int,
) -> Tuple[ModelInterfaceInProcess, SingleVectorViz, MatricesLabel]:
    """
    Boilerplate setup function to get a few common things in place for visualization.
    :param model_not_matching_ok: Process this arg.
    :param projection_model_path: Path to the model to use in subsequent projections.
    :param projection_file_reader: File being read in visualization.
    :param video_height: Visualization will be a square.
    :return: Configured elements to be consumed by visualizations.
    """

    if not model_not_matching_ok:
        input_model_hash = hash_file(projection_model_path)

        if input_model_hash != projection_file_reader.projection_attributes.model_md5_hash:
            raise ValueError("Input model was not the one used in projection.")

    model = create_model_interface_process(model_path=projection_model_path)

    # This isn't consumed for visualization, but is used to understand properties of the
    # latent histories.
    matrices_label = final_latents_matrices_label(projection_file_reader)

    make_visualization = vector_visualizer(
        y_min=matrices_label.data.min(),
        y_max=matrices_label.data.max(),
        title=matrices_label.label,
        output_width=video_height,
        output_height=video_height,
    )

    return model, make_visualization, matrices_label


def visualize_projection_history(  # pylint: disable=too-many-locals
    projection_file_path: Path,
    output_video_path: Path,
    projection_model_path: Path,
    model_not_matching_ok: bool,
    video_height: Optional[int] = 1024,
) -> None:
    """
    Shows each projection history latent over time as it approaches the target.

    This function uses the input model to visualize each projection set. However if the projection
    file was created with these `image_histories` recorded, you wouldn't need to re-synthesize.
    TODO: detect if synthesis step is needed.
    :param projection_file_path: Path to file to read.
    :param output_video_path: The path to write the resulting video to.
    :param projection_model_path: Path to the model to re-create the projection.
    :param model_not_matching_ok: If the input model given by `projection_model_path` doesn't
    match the model that was used in the projection file, raise a ValueError if given.
    :param video_height: The height of the output video in pixels, the width will be 3x the height.
    :return: None
    """

    with load_projection_file(projection_file_path) as reader:

        model, make_visualization, matrices_label = _setup_visualization(
            model_not_matching_ok=model_not_matching_ok,
            projection_model_path=projection_model_path,
            projection_file_reader=reader,
            video_height=video_height,
        )

        video = create_video_writer(
            video_path=output_video_path,
            num_squares=3,
            video_fps=reader.projection_attributes.projection_fps,
            video_height=video_height,
        )

        x_values = np.arange(matrices_label.vector_length)

        for projection_frame_index, (latent_history, target) in enumerate(
            zip(reader.latents_histories, reader.target_images)
        ):

            for latent_index, latents in enumerate(latent_history):

                projection_image = model.model_interface.create_image_matrix(latents)

                with make_visualization(
                    x_values=x_values,
                    y_values=latents,
                    new_title=(
                        f"{matrices_label.label} frame: {projection_frame_index}, "
                        f"step: {latent_index}"
                    ),
                ) as visualization:
                    # Puts the data visualization to the left of the synthesis.
                    frame = cv2.hconcat([visualization, projection_image, target])

                video.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

                LOGGER.info(
                    f"Wrote frame: {output_video_path.name}, "
                    f"projection frame: {projection_frame_index + 1}, "
                    f"latent step: {latent_index}"
                )

        model.stop_function()
        video.release()


def visualize_partial_projection_history(  # pylint: disable=too-many-locals
    projection_file_path: Path,
    output_video_path: Path,
    projection_model_path: Path,
    model_not_matching_ok: bool,
    projection_step_to_take: int,
    video_height: Optional[int] = 1024,
) -> None:
    """
    Shows the effect that using a latent from the projection history rather than the final latents
    to create the output images has visually. Creates a video that shows the target, next to the
    synthesized final latents, next to the synthesis output from the latent plucked from the
    projection history.

    This function uses the input model to visualize each projection set. However if the projection
    file was created with these `image_histories` recorded, you wouldn't need to re-synthesize.
    TODO: detect if synthesis step is needed.
    :param projection_file_path: Path to file to read.
    :param output_video_path: The path to write the resulting video to.
    :param projection_model_path: Path to the model to re-create the projection.
    :param model_not_matching_ok: If the input model given by `projection_model_path` doesn't
    match the model that was used in the projection file, raise a ValueError if given.
    :param projection_step_to_take: This step of projection will be retrieved and fed to the model.
    :param video_height: The height of the output video in pixels, the width will be 3x the height.
    :return: None
    """

    with load_projection_file(projection_file_path) as reader:

        model, make_visualization, matrices_label = _setup_visualization(
            model_not_matching_ok=model_not_matching_ok,
            projection_model_path=projection_model_path,
            projection_file_reader=reader,
            video_height=video_height,
        )

        video = create_video_writer(
            video_path=output_video_path,
            num_squares=4,
            video_fps=1,
            video_height=video_height,
        )

        x_values = np.arange(matrices_label.vector_length)

        for projection_frame_index, (latent_history, target, final) in enumerate(
            zip(reader.latents_histories, reader.target_images, reader.final_images)
        ):

            latents = next(itertools.islice(latent_history, projection_step_to_take, None))

            projection_image = model.model_interface.create_image_matrix(latents)

            with make_visualization(x_values=x_values, y_values=latents) as visualization:
                # Puts the data visualization to the left of the synthesis.
                frame = cv2.hconcat([visualization, projection_image, target, final])

            video.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

            LOGGER.info(
                f"Wrote frame: {output_video_path.name}, "
                f"projection frame: {projection_frame_index + 1}, "
                f"latent step: {projection_step_to_take}"
            )

        model.stop_function()
        video.release()
