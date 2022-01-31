"""
Functions to feed vectors into a model and record the output.
Also tools to visualize these vectors against the model outputs.
"""

import logging
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import click
from click_option_group import RequiredMutuallyExclusiveOptionGroup, optgroup

from gance.assets import OUTPUT_DIRECTORY
from gance.data_into_model_visualization.model_visualization import viz_model_ins_outs
from gance.data_into_model_visualization.visualization_inputs import (
    CreateVisualizationInput,
    alpha_blend_projection_file,
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.image_sources import video_common
from gance.logger_common import LOGGER
from gance.model_interface.model_functions import MultiModel, sorted_models_in_directory
from gance.projection import projection_file_reader
from gance.vector_sources.music import read_wav_scale_for_video

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


def _create_visualization(
    audio_path: Path,
    models: Optional[MultiModel],
    vector_function: CreateVisualizationInput,
    vector_length: int,
    video_fps: float,
    output_path: Path,
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
    :param output_path: The path to write the output video.
    :param enable_3d: If True, a 3d visualization of the input vectors will be created alongside
    the output of the model (if the `models` is not None).
    :param enable_2d: If True, a 2d visualization of the combination of the input vectors
    will be created alongside the output of the model.
    :param frames_to_visualize: The number of frames in the input to visualize. Starts from the
    first index, goes to this value. Ex. 10 is given, the first 10 frames will be visualized.
    :return: The path to the output video.
    """

    LOGGER.info(f"Writing video: {output_path}")

    time_series_audio_vectors = read_wav_scale_for_video(
        wav=audio_path,
        vector_length=vector_length,
        frames_per_second=video_fps,
    ).wav_data

    model_output = viz_model_ins_outs(
        models=models,
        data=vector_function(
            time_series_audio_vectors=time_series_audio_vectors,
            vector_length=vector_length,
            model_indices=models.model_indices if models is not None else [0, 1, 2],  # TODO
        ),
        default_vector_length=vector_length,
        enable_3d=enable_3d,
        enable_2d=enable_2d,
        frames_to_visualize=frames_to_visualize,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:

        tmp_video_path = Path(f.name)

        frames = video_common.horizontal_concat_optional_sources(
            [model_output.visualization_images, model_output.model_images]
        )

        video_common.write_source_to_disk_consume(
            source=frames,
            video_path=tmp_video_path,
            video_fps=video_fps,
        )

        f.flush()

        while not tmp_video_path.exists():
            pass

        video_common.add_wav_to_video(
            video_path=tmp_video_path,
            audio_path=audio_path,
            output_path=output_path,
        )

    return output_path


def _configure_run(
    wav: str,
    output_path: str,
    models_directory: Optional[str],
    vector_length: Optional[int],
    index: Optional[Tuple[int, ...]],
    frames_to_visualize: Optional[int],
    output_fps: float,
    debug_2d: bool,
    vector_function: CreateVisualizationInput,
) -> None:
    """
    Coerce UI Input
    :param wav: See click docs.
    :param output_path: See click docs.
    :param models_directory: See click docs.
    :param vector_length: See click docs.
    :param index: See click docs.
    :param frames_to_visualize: See click docs.
    :param output_fps: See click docs.
    :param debug_2d: See click docs.
    :param vector_function: Actual function to run.
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
            output_path=Path(output_path),
            enable_3d=False,  # TODO: Make this a CLI argument
            enable_2d=debug_2d,
            frames_to_visualize=frames_to_visualize,
        )


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
                "--output_path",
                help="Output video will be written to this path",
                type=click.Path(
                    exists=False,
                    file_okay=True,
                    writable=True,
                    dir_okay=False,
                    resolve_path=True,
                ),
                default=str(Path(OUTPUT_DIRECTORY.joinpath("output.mp4"))),
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
            click.option(
                "--alpha",
                help="Alpha blending coefficient.",
                type=click.FloatRange(min=0),
                required=False,
                default=0.5,
                show_default=True,
            ),
            click.option(
                "--fft_roll_enabled",
                help="If true, the FFT vectors move over time.",
                is_flag=True,
                required=False,
                default=False,
                show_default=True,
            ),
            click.option(
                "--fft_amplitude_range",
                help="Values in FFT are scaled to this range.",
                type=click.Tuple(types=(float, float)),
                required=False,
                default=(-4, 4),
                show_default=True,
            ),
        ]
    ):
        func = option(func)

    return func


@click.group()  # pylint: disable=too-many-arguments
def cli() -> None:
    """
    Use one of the commands below to project either a list of individual videos, or a directory
    of videos.

    \f

    :return: None
    """


@cli.command()  # type: ignore
@common_command_options
def noise_blend(  # pylint: disable=too-many-arguments
    wav: str,
    output_path: str,
    models_directory: Optional[str],
    vector_length: Optional[int],
    index: Optional[Tuple[int, ...]],
    frames_to_visualize: Optional[int],
    output_fps: float,
    debug_2d: bool,
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
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

    _configure_run(
        wav=wav,
        output_path=output_path,
        models_directory=models_directory,
        vector_length=vector_length,
        index=index,
        frames_to_visualize=frames_to_visualize,
        output_fps=output_fps,
        debug_2d=debug_2d,
        vector_function=partial(
            alpha_blend_vectors_max_rms_power_audio,
            alpha,
            fft_roll_enabled,
            fft_amplitude_range,
        ),
    )


@cli.command()  # type: ignore   # pylint: disable=too-many-arguments
@common_command_options
@click.option(
    "--projection_file_path",
    help="Path to the projection file.",
    type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True),
    required=True,
)
@click.option(
    "--blend_depth",
    help=(
        "Number of vectors within the final latents matrices that receive the FFT during "
        "alpha blending."
    ),
    type=click.IntRange(min=0, max=18),
    required=False,
    default=0.5,
    show_default=True,
)
def projection_file_blend(  # pylint: disable=too-many-arguments
    wav: str,
    output_path: str,
    models_directory: Optional[str],
    vector_length: Optional[int],
    index: Optional[Tuple[int, ...]],
    frames_to_visualize: Optional[int],
    output_fps: float,
    debug_2d: bool,
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    projection_file_path: str,
    blend_depth: int,
) -> None:
    """
    Visualize one or more wav files using StyleGAN2 models, save the visualizations as videos.

    \f
    :param wav: Path to the wav file to input to the model(s).
    :param output_path: "Output files (videos, images etc) will be written to this directory.
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

    with projection_file_reader.load_projection_file(Path(projection_file_path)) as reader:

        vector_function = partial(
            alpha_blend_projection_file,
            projection_file_reader.final_latents_matrices_label(reader),
            alpha,
            fft_roll_enabled,
            fft_amplitude_range,
            blend_depth,
        )

        _configure_run(
            wav=wav,
            output_path=output_path,
            models_directory=models_directory,
            vector_length=vector_length,
            index=index,
            frames_to_visualize=frames_to_visualize,
            output_fps=output_fps,
            debug_2d=debug_2d,
            vector_function=vector_function,
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
