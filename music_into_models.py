"""
Functions to feed vectors into a model and record the output.
Also tools to visualize these vectors against the model outputs.
"""

import datetime
import inspect
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import click
from click_option_group import RequiredMutuallyExclusiveOptionGroup, optgroup

from gance.assets import OUTPUT_DIRECTORY
from gance.cli_common import EXTENSION_MP4
from gance.data_into_model_visualization.model_visualization import viz_model_ins_outs
from gance.data_into_model_visualization.visualization_common import CreateVisualizationInput
from gance.data_into_model_visualization.visualization_inputs import alpha_blend_projection_file
from gance.logger_common import LOGGER
from gance.model_interface.model_functions import MultiModel, sorted_models_in_directory
from gance.vector_sources.music import read_wav_scale_for_video
from gance.video_common import add_wav_to_video

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


def _create_visualization(
    audio_path: Path,
    models: Optional[MultiModel],
    vector_function: CreateVisualizationInput,
    vector_length: int,
    video_fps: float,
    output_directory: Path,
    enable_3d: bool,
    enable_2d: bool,
    frames_to_visualize: Optional[int] = None,
) -> Path:
    """
    Create a video visualization of a given audio file. The video will be of three different
    visualizations side by side.
    The left visualization is the "moving dot" 3d vector visualization, showing where the current
    vector is relative to all of the vectors in the video.
    The center visualization is a 2D scatter plot of the current vector.
    If the a model is given, the right visualization is the output of the model given the current
    input vector.
    The audio is scaled to be able to create a single vector for every frame in the video.
    The input audio is added to the video file.
    :param audio_path: The audio file to visualize.
    :param models: The model to input vectors to.
    :param vector_function: The function to apply to the audio data before feeding it into the
    model. Think an spectrogram conversion or a smoothing function.
    :param vector_length: The side length of the model shape, the number of points per vector.
    :param video_fps: The FPS of the output video.
    :param output_directory: The directory to write the output video.
    :param enable_3d: If True, a 3d visualization of the input vectors will be created alongside
    the output of the model (if the `models` is not None).
    :param enable_2d: If True, a 2d visualization of the combination of the input vectors
    will be created alongside the output of the model.
    :param frames_to_visualize: The number of frames in the input to visualize. Starts from the
    first index, goes to this value. Ex. 10 is given, the first 10 frames will be visualized.
    :return: The path to the output video.
    """

    video_path = output_directory.joinpath(f"./model_vis_output_{audio_path.name}.{EXTENSION_MP4}")

    LOGGER.info(f"Writing video: {video_path}")

    time_series_audio_vectors = read_wav_scale_for_video(
        audio_path, vector_length, video_fps
    ).wav_data

    video_path = viz_model_ins_outs(
        models=models,
        data=vector_function(
            time_series_audio_vectors=time_series_audio_vectors,
            vector_length=vector_length,
            model_indices=models.model_indices if models is not None else [0, 1, 2],  # TODO
        ),
        output_video_path=video_path,
        default_vector_length=vector_length,
        video_fps=video_fps,
        enable_3d=enable_3d,
        enable_2d=enable_2d,
        frames_to_visualize=frames_to_visualize,
    )

    video_with_audio_path = output_directory.joinpath(
        f"{video_path.with_suffix('').name}_with_audio{video_path.suffix}"
    )

    while not video_path.exists():
        pass

    add_wav_to_video(
        video_path=video_path,
        audio_path=audio_path,
        output_path=video_with_audio_path,
    )

    return video_with_audio_path


def common_command_options(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Create a decorator that contains the common options seen in both commands.
    :param func: Function to decorate.
    :return: Decorated function.
    """

    for option in reversed(
        [
            click.option(  # TODO: add a note about the format of the wav file here.
                "--wav",
                help="Path to the wav file to input to the model(s).",
                type=click.Path(
                    exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True
                ),
                required=True,
            ),
            click.option(
                "--output_directory",
                help="Output files (videos, images etc) will be written to this directory.",
                type=click.Path(
                    exists=False,
                    file_okay=False,
                    readable=True,
                    writable=True,
                    dir_okay=True,
                    resolve_path=True,
                ),
                default=str(
                    Path(
                        OUTPUT_DIRECTORY.joinpath(
                            f'output_{datetime.datetime.now().strftime("%m-%d-%Y-%H_%M_%S")}'
                        )
                    )
                ),
                show_default=True,
            ),
            optgroup.group(
                "Model sources",
                cls=RequiredMutuallyExclusiveOptionGroup,
                help=(
                    "Must provide a directory that contains models, or a substitute vector length "
                    "to just do numeric visualizations"
                ),
            ),
            optgroup.option(
                "--models_directory",
                help="Model `.pkl` files will be read from this directory.",
                type=click.Path(
                    exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True
                ),
                default=None,
            ),
            optgroup.option(
                "--vector_length",
                type=click.IntRange(min=1),
                default=None,
                help="Numeric visualizations will use this number",
            ),
            click.option(
                "--index",
                help=(
                    "If given, only models from these indices "
                    "(locations in the sorted list of models in the input dir) will be used."
                ),
                type=click.IntRange(min=0),
                multiple=True,
                default=None,
                required=False,
                show_default=True,
            ),
            click.option(
                "--frames_to_visualize",
                help=(
                    "The number of frames in the input to visualize. "
                    "Starts from the first index, goes to this value. "
                    "Ex. 10 is given, the first 10 frames will be visualized."
                ),
                type=click.IntRange(min=0),
                default=None,
                required=False,
                show_default=True,
            ),
            click.option(
                "--output_fps",
                help=(
                    "Frames per second of output video. Input sources will be upscaled to "
                    "satisfy this requirement."
                ),
                type=click.FloatRange(min=0),
                required=False,
                default=60,
                show_default=True,
            ),
            click.option(
                "--debug_2d",
                help=(
                    "A visualization of the input sources that were combined as the input to the "
                    "model will be rendered and placed in the output."
                ),
                is_flag=True,
                required=False,
                default=False,
                show_default=True,
            ),
        ]
    ):
        func = option(func)

    return func


@click.group()
def cli() -> None:
    """
    Use one of the commands below to project either a list of individual videos, or a directory
    of videos.

    \f

    :return: None
    """


@cli.command()  # type: ignore
@common_command_options
@click.option(
    "--projection_file_path",
    help="Path to the projection file.",
    type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True),
    required=True,
)
@click.option(
    "--alpha",
    help="Alpha blending coefficient.",
    type=click.FloatRange(min=0),
    required=False,
    default=0.5,
    show_default=True,
)
def bb(
    wav: str,
    output_directory: str,
    models_directory: Optional[str],
    vector_length: Optional[int],
    index: Optional[Tuple[int, ...]],
    frames_to_visualize: Optional[int],
    output_fps: float,
    projection_file_path: str,
    alpha: float,
) -> None:
    pass


def projection_file(
    wav: str,
    output_directory: str,
    models_directory: Optional[str],
    vector_length: Optional[int],
    index: Optional[Tuple[int, ...]],
    frames_to_visualize: Optional[int],
    output_fps: float,
    debug_2d: bool,
    note: Optional[str],
    projection_file_path: str,
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    fft_depth: int,
) -> None:
    """
    Visualize one or more wav files using StyleGAN2 models, save the visualizations as videos.

    \f
    :param wav: Path to the wav file to input to the model(s).
    :param output_directory: "Output files (videos, images etc) will be written to this directory.
    :param models_directory: Model `.pkl` files will be read from this directory.
    :param index: If given, only models from these indices (locations in the sorted list of
    models in the input dir) will be used.
    :param frames_to_visualize: The number of frames in the input to visualize. Starts from the
    first index, goes to this value. Ex. 10 is given, the first 10 frames will be visualized.
    :param output_fps: Frames per second of output video. Input sources will be upscaled to
    satisfy this requirement.
    :param projection_file_path: See click help.
    :param alpha: See click help.
    :return: None
    """

    # Get the paths to the models to be used.
    if models_directory is not None:
        models_directory_path = Path(models_directory)
        all_models = sorted_models_in_directory(models_directory=models_directory_path)
        if not all_models:
            raise ValueError(f"No models found in directory {models_directory_path}")
        model_paths = (
            [all_models[model_index] for model_index in sorted(index)] if index else all_models
        )
    else:
        model_paths = []

    # Set up the output directory, multiple files could get written here.
    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(exist_ok=True)

    with open(output_directory_path.joinpath("visualization_input_function.txt"), "w") as f:
        f.writelines(inspect.getsource(alpha_blend_projection_file))
        f.writelines([f"Alpha: {alpha}", note])

    vector_function = partial(
        alpha_blend_projection_file,
        Path(projection_file_path),
        alpha,
        fft_roll_enabled,
        fft_amplitude_range,
        fft_depth,
    )

    with MultiModel(model_paths=model_paths) as multi_models:

        if multi_models is None:
            input_vector_length = vector_length
        else:
            input_vector_length = multi_models.expected_vector_length

        # Throw away the output here, don't need to persist.
        _create_visualization(
            audio_path=Path(wav),
            models=multi_models,
            vector_function=vector_function,
            vector_length=input_vector_length,
            video_fps=output_fps,
            output_directory=output_directory_path,
            enable_3d=False,  # TODO: Make this a CLI argument
            enable_2d=debug_2d,  # TODO: Make this a CLI argument
            frames_to_visualize=frames_to_visualize,
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
