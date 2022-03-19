"""
Feed inputs (music, videos) into a network and record the output.
Also tools to visualize these vectors against the network outputs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import click
import more_itertools
import numpy as np
import pandas as pd
from click_option_group import AllOptionGroup, RequiredAnyOptionGroup, optgroup
from cv2 import cv2
from lz.transposition import transpose

from gance import divisor, overlay
from gance.assets import OUTPUT_DIRECTORY
from gance.data_into_network_visualization import visualize_vector_reduction
from gance.data_into_network_visualization.network_visualization import vector_synthesis
from gance.data_into_network_visualization.visualization_inputs import (
    alpha_blend_projection_file,
    alpha_blend_vectors_max_rms_power_audio,
)
from gance.gance_types import RGBInt8ImageType
from gance.image_sources import video_common
from gance.iterator_on_disk import HDF5_SERIALIZER, iterator_on_disk
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import MultiNetwork, parse_network_paths
from gance.projection import projection_file_reader
from gance.vector_sources import music, vector_reduction
from gance.vector_sources.vector_reduction import DataLabel, ResultLayers
from gance.vector_sources.vector_sources_common import underlying_length
from gance.vector_sources.vector_types import ConcatenatedVectors

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


@click.group()
def cli() -> None:
    """
    Feed audio files into StyleGAN2 networks. Saves the resulting synthesized images into a
    video scored by the input audio.

    Each subcommand exposes an option to create a visualization of the transforms on the input
    data, to understand how each of the parameters drive the results.

    \f

    :return: None
    """


def common_command_options(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Create a decorator that contains the common options seen in both commands.
    :param func: Function to decorate.
    :return: Decorated function.
    """

    for option in reversed(
        [
            click.option(
                "-a",
                "--wav",
                help="Path to the wav file to input to the network(s).",
                type=click.Path(
                    exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True
                ),
                required=True,
                multiple=True,
            ),
            click.option(
                "-o",
                "--output-path",
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
                "network sources",
                cls=RequiredAnyOptionGroup,
                help=(
                    "Must provide a directory that contains networks, or paths to "
                    "specific networks."
                ),
            ),
            optgroup.option(
                "-d",
                "--networks-directory",
                help=(
                    "network `.pkl` files will be read from this directory. "
                    "These will be alphanumerically sorted."
                ),
                type=click.Path(
                    exists=True, file_okay=False, readable=True, dir_okay=True, resolve_path=True
                ),
                default=None,
            ),
            optgroup.option(
                "-m",
                "--network-path",
                help=(
                    "Paths to particular network files. "
                    "These will be used for synthesis in the order they're given."
                ),
                type=click.Path(
                    exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True
                ),
                multiple=True,
                default=None,
            ),
            optgroup.option(
                "--networks-json",
                help=(
                    'Path to a JSON file with a single key: "networks" '
                    "that maps to a list of paths "
                    "to network pickle files. These will be used for synthesis in the order they "
                    "appear in the file."
                ),
                type=click.Path(
                    exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True
                ),
                default=None,
            ),
            click.option(
                "-n",
                "--frames-to-visualize",
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
                "-f",
                "--output-fps",
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
                "-s",
                "--output-side-length",
                help=(
                    "Both the synthesized output from the network, and the individual elements of "
                    "the debug visualizations are squares. This sets their side lengths in pixels."
                ),
                type=click.IntRange(min=1),
                required=False,
                default=1024,
                show_default=True,
            ),
            optgroup.group(
                "Debug Visualization Parameters",
                cls=AllOptionGroup,
                help="Control the visualization produced alongside the output video.",
            ),
            optgroup.option(
                "--debug-path",
                help=(
                    "If provided, a video containing debug visualizations of the "
                    "synthesis steps will be written to this path."
                ),
                type=click.Path(
                    exists=False, file_okay=True, writable=True, dir_okay=False, resolve_path=True
                ),
            ),
            optgroup.option(
                "--debug-window",
                help=(
                    "For visualizations that represent trends in data that span multiple "
                    "frames of video, this param controls the number of frames to represent "
                    "per frame. Ex: 100"
                ),
                type=click.IntRange(min=0),
            ),
            click.option(
                "--alpha",
                help="Alpha blending coefficient.",
                type=click.FloatRange(min=0),
                required=False,
                default=0.25,
                show_default=True,
            ),
            click.option(
                "--fft-roll-enabled",
                help="If true, the FFT vectors move over time.",
                is_flag=True,
                required=False,
                show_default=True,
            ),
            click.option(
                "--fft-amplitude-range",
                help="Values in FFT are scaled to this range.",
                type=click.Tuple(types=(float, float)),
                required=False,
                default=(-1, 1),
                show_default=True,
            ),
            click.option(
                "--run-config",
                help="If given, a JSON file containing input parameters, and metadata about the "
                "output videos is written to this path after the run has finished.",
                type=click.Path(
                    exists=False, file_okay=True, writable=True, dir_okay=False, resolve_path=True
                ),
                required=False,
                default=None,
                show_default=True,
            ),
        ]
    ):
        func = option(func)

    return func


def write_input_args(
    output_path: Optional[str],
    input_locals: Dict[str, Any],
    network_paths: Optional[List[Path]] = None,
) -> None:
    """
    Function to dump input args to CLI function to a file.
    :param output_path: Path to the file to write, don't do anything if this value is None.
    :param input_locals: The `locals()` call right after the CLI is invoked.
    :return: None
    """

    if output_path is not None:

        if network_paths is None:
            multi_networks: Optional[MultiNetwork] = input_locals.pop("multi_networks", None)
            network_paths = multi_networks.network_paths if multi_networks else None

        with open(output_path, "w") as f:
            json.dump(
                {"current_locals": input_locals, "networks_paths": [str(p) for p in network_paths]},
                f,
                indent=4,
            )


@cli.command()  # type: ignore
@common_command_options
def noise_blend(  # pylint: disable=too-many-arguments,too-many-locals
    wav: List[str],
    output_path: str,
    networks_directory: Optional[str],
    network_path: Optional[List[str]],
    networks_json: Optional[str],
    frames_to_visualize: Optional[int],
    output_fps: float,
    output_side_length: int,
    debug_path: Optional[str],
    debug_window: Optional[int],
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    run_config: Optional[str],
) -> None:
    """
    Transform audio data, combine it with smoothed random noise, and feed the result into a network
    for synthesis.

    \f
    :param wav: See click help.
    :param output_path: See click help.
    :param networks_directory: See click help.
    :param network_path: See click help.
    :param networks_json: See click help.
    :param frames_to_visualize: See click help.
    :param output_fps: See click help.
    :param output_side_length: See click help.
    :param debug_path: See click help.
    :param debug_window: See click help.
    :param alpha: See click help.
    :param fft_roll_enabled: See click help.
    :param fft_amplitude_range: See click help.
    :param run_config: See click help.
    :return: None
    """

    input_locals = locals()
    network_paths = parse_network_paths(
        networks_directory=networks_directory, networks=network_path, networks_json=networks_json
    )

    write_input_args(output_path=run_config, input_locals=input_locals, network_paths=network_paths)

    audio_paths = list(map(Path, wav))

    with MultiNetwork(network_paths=network_paths) as multi_networks:

        LOGGER.info(f"Writing video: {output_path}")

        time_series_audio_vectors = cast(
            ConcatenatedVectors,
            music.read_wavs_scale_for_video(
                wavs=audio_paths,
                vector_length=multi_networks.expected_vector_length,
                frames_per_second=output_fps,
            ).wav_data,
        )

        synthesis_output = vector_synthesis(
            networks=multi_networks,
            data=alpha_blend_vectors_max_rms_power_audio(
                alpha=alpha,
                fft_roll_enabled=fft_roll_enabled,
                fft_amplitude_range=fft_amplitude_range,
                time_series_audio_vectors=time_series_audio_vectors,
                vector_length=multi_networks.expected_vector_length,
                network_indices=multi_networks.network_indices,
            ),
            default_vector_length=multi_networks.expected_vector_length,
            enable_3d=False,
            enable_2d=debug_path is not None,
            frames_to_visualize=frames_to_visualize,
            network_index_window_width=debug_window,
            video_height=output_side_length,
        )

        forwarded_hero_frames = video_common.write_source_to_disk_forward(
            source=synthesis_output.synthesized_images,
            video_path=Path(output_path),
            video_fps=output_fps,
            audio_paths=audio_paths,
        )

        if synthesis_output.visualization_images is not None and debug_path is not None:
            video_common.write_source_to_disk_consume(
                source=(
                    cv2.hconcat(*frames)
                    for frames in (
                        zip(forwarded_hero_frames, synthesis_output.visualization_images)
                    )
                ),
                video_path=Path(debug_path),
                video_fps=output_fps,
                audio_paths=audio_paths,
            )
        else:
            # This causes the video to be written.
            more_itertools.consume(forwarded_hero_frames)


def _projection_file_blend_api(  # pylint: disable=too-many-arguments,too-many-locals
    wav: List[str],
    output_path: str,
    multi_networks: MultiNetwork,
    frames_to_visualize: Optional[int],
    output_fps: float,
    output_side_length: int,
    debug_path: Optional[str],
    debug_window: Optional[int],
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    run_config: Optional[str],
    projection_file_path: str,
    blend_depth: int,
    complexity_change_rolling_sum_window: int,
    complexity_change_threshold: int,
    phash_distance: int,
    bbox_distance: float,
    track_length: int,
) -> None:
    """
    See docs in CLI function.
    """

    input_locals = locals()

    write_input_args(output_path=run_config, input_locals=input_locals)

    create_debug_visualization = debug_path is not None

    audio_paths = list(map(Path, wav))

    with projection_file_reader.load_projection_file(Path(projection_file_path)) as reader:

        final_latents = projection_file_reader.final_latents_matrices_label(reader)

        final_latents_in_file = (
            underlying_length(final_latents.data) / multi_networks.expected_vector_length
        )
        processed_frames_in_file = reader.projection_attributes.projection_frame_count
        projection_complete = reader.projection_attributes.complete

        LOGGER.info(
            f"Reading projection file. Complete: {projection_complete}, "
            f"Final Latent Count: {final_latents_in_file}, "
            f"Processed Frames: {processed_frames_in_file}"
        )

        if not projection_complete or abs(final_latents_in_file - processed_frames_in_file) > 2:
            raise ValueError("Invalid Projection File, cannot continue.")

        frame_multiplier = divisor.divide_no_remainder(
            numerator=output_fps,
            denominator=reader.projection_attributes.projection_fps,
        )

        time_series_audio_vectors = cast(
            ConcatenatedVectors,
            music.read_wavs_scale_for_video(
                wavs=audio_paths,
                vector_length=multi_networks.expected_vector_length,
                target_num_vectors=int(frame_multiplier * final_latents_in_file),
            ).wav_data,
        )

        synthesis_output = vector_synthesis(
            networks=multi_networks,
            data=alpha_blend_projection_file(
                final_latents_matrices_label=final_latents,
                alpha=alpha,
                fft_roll_enabled=fft_roll_enabled,
                fft_amplitude_range=fft_amplitude_range,
                blend_depth=blend_depth,
                time_series_audio_vectors=time_series_audio_vectors,
                vector_length=multi_networks.expected_vector_length,
                network_indices=multi_networks.network_indices,
            ),
            default_vector_length=multi_networks.expected_vector_length,
            enable_3d=False,
            enable_2d=create_debug_visualization,
            frames_to_visualize=frames_to_visualize,
            network_index_window_width=debug_window,
            video_height=output_side_length,
        )

        music_complexity_overlay_mask = vector_reduction.rolling_sum_results_layers(
            vector_reduction.absolute_value_results_layers(
                results_layers=ResultLayers(
                    result=DataLabel(
                        data=vector_reduction.derive_results_layers(
                            vector_reduction.reduce_vector_gzip_compression_rolling_average(
                                time_series_audio_vectors=time_series_audio_vectors,
                                vector_length=multi_networks.expected_vector_length,
                            ),
                            order=1,
                        ).result.data,
                        label="Gzipped audio, smoothed, averaged, 1st order derivation.",
                    ),
                ),
            ),
            window_length=complexity_change_rolling_sum_window,
        )

        skip_mask: List[bool] = list(
            pd.Series(music_complexity_overlay_mask.result.data).fillna(np.inf)
            > complexity_change_threshold
        )

        foreground_iterators = iter(
            iterator_on_disk(
                iterator=more_itertools.repeat_each(
                    reader.target_images,
                    frame_multiplier,
                ),
                copies=1,
                serializer=HDF5_SERIALIZER,
            )
        )

        background_iterators = iter(
            iterator_on_disk(
                iterator=synthesis_output.synthesized_images,
                copies=1,
                serializer=HDF5_SERIALIZER,
            )
        )

        overlay_results = overlay.overlay_eye_tracking.compute_eye_tracking_overlay(
            foreground_images=next(foreground_iterators),
            background_images=next(background_iterators),
            min_phash_distance=phash_distance,
            min_bbox_distance=bbox_distance,
            skip_mask=skip_mask,
        )

        logging.info("Starting to compute mask to filter out short sequences of overlay frames.")

        boxes_list = list(overlay_results.bbox_lists)

        long_tracks_mask = vector_reduction.track_length_filter(
            bool_tracks=(
                ~pd.Series(skip_mask) & pd.Series((box is not None for box in boxes_list))
            ),
            track_length=track_length,
        )

        final_frames: Iterator[Tuple[RGBInt8ImageType, RGBInt8ImageType]] = (
            (
                overlay.overlay_common.write_boxes_onto_image(
                    foreground_image=foreground,
                    background_image=background,
                    bounding_boxes=bounding_boxes,
                )
                if in_long_track
                else background,
                foreground,
            )
            for (bounding_boxes, foreground, background, in_long_track) in zip(
                boxes_list,
                next(foreground_iterators),
                next(background_iterators),
                long_tracks_mask,
            )
        )

        finals, foregrounds = transpose(final_frames)

        finals = video_common.write_source_to_disk_forward(
            source=finals,
            video_path=Path(output_path),
            video_fps=output_fps,
            audio_paths=audio_paths,
        )

        if create_debug_visualization:

            visualization = overlay.overlay_visualization.visualize_overlay_computation(
                overlay=overlay_results.contexts,
                frames_per_context=debug_window,
                video_square_side_length=output_side_length,
                horizontal_lines=overlay.overlay_visualization.VisualizeOverlayThresholds(
                    phash_line=phash_distance, bbox_distance_line=bbox_distance
                ),
            )

            music_overlay_mask_visualization = visualize_vector_reduction.visualize_result_layers(
                result_layers=music_complexity_overlay_mask,
                frames_per_context=debug_window,
                video_height=output_side_length,
                title="Overlay binary mask",
                horizontal_line=complexity_change_threshold,
            )

            video_common.write_source_to_disk_consume(
                source=(
                    cv2.vconcat(
                        [
                            cv2.hconcat(
                                [
                                    final,
                                    background,
                                    foreground,
                                ]
                            ),
                            cv2.hconcat(
                                [
                                    music_overlay_mask_visualization_image,
                                    overlay_visualization_frame,
                                    visualization_image,
                                ]
                            ),
                        ]
                    )
                    for (
                        final,
                        foreground,
                        background,
                        overlay_visualization_frame,
                        visualization_image,
                        music_overlay_mask_visualization_image,
                    ) in zip(
                        finals,
                        foregrounds,
                        more_itertools.repeat_each(
                            reader.final_images,
                            frame_multiplier,
                        ),
                        visualization,
                        synthesis_output.visualization_images,
                        music_overlay_mask_visualization,
                    )
                ),
                video_path=Path(debug_path),
                video_fps=output_fps,
                audio_paths=audio_paths,
            )
        else:
            more_itertools.consume(finals)


@cli.command()  # pylint: disable=too-many-arguments
@common_command_options
@click.option(
    "--projection-file-path",
    help="Path to the projection file.",
    type=click.Path(exists=True, file_okay=True, readable=True, dir_okay=False, resolve_path=True),
    required=True,
)
@click.option(
    "--blend-depth",
    help=(
        "Number of vectors within the final latents matrices that receive the FFT during "
        "alpha blending."
    ),
    type=click.IntRange(min=0, max=18),
    required=False,
    default=10,
    show_default=True,
)
@optgroup.group(
    "Eye-Tracking Overlay Parameters",
    cls=AllOptionGroup,
    help=(
        "Controls how eye-containing sections of the target images get overlaid on "
        "top of the output images. "
    ),
)
@optgroup.option(
    "-w",
    "--complexity-change-rolling-sum-window",
    type=click.IntRange(min=0),
    help="The number of frames to window the music complexity computation to.",
    default=30,
    show_default=True,
)
@optgroup.option(
    "-t",
    "--complexity-change-threshold",
    type=click.IntRange(min=0),
    help="If complexity is under this value, an overlay computation is enabled.",
    default=100,
    show_default=True,
)
@optgroup.option(
    "-p",
    "--phash-distance",
    type=click.IntRange(min=0),
    help=(
        "Minimum distance between perceptual hashes of the bounding box region of the synthesized "
        "image and its corresponding target frame to enable an overlay computation. "
    ),
    default=30,
    show_default=True,
)
@optgroup.option(
    "-b",
    "--bbox-distance",
    type=click.FloatRange(min=0),
    help=(
        "For pairs of synthesized images and their corresponding targets that both contain eye "
        "bounding boxes, this value is the minimum distance in pixels between "
        "the origins of those bounding boxes to enable an overlay computation."
    ),
    default=100,
    show_default=True,
)
@optgroup.option(
    "-t",
    "--track-length",
    type=click.IntRange(min=0),
    help=(
        "For sequences of adjacent frames that could contain an overlay, "
        "this parameter is the minimum number of overlay frames in a row to be included in the "
        "output video."
    ),
    default=10,
    show_default=True,
)
def projection_file_blend(  # pylint: disable=too-many-arguments,too-many-locals
    wav: List[str],
    output_path: str,
    networks_directory: Optional[str],
    network_path: Optional[List[str]],
    networks_json: Optional[str],
    frames_to_visualize: Optional[int],
    output_fps: float,
    output_side_length: int,
    debug_path: Optional[str],
    debug_window: Optional[int],
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    run_config: Optional[str],
    projection_file_path: str,
    blend_depth: int,
    complexity_change_rolling_sum_window: int,
    complexity_change_threshold: int,
    phash_distance: int,
    bbox_distance: float,
    track_length: int,
) -> None:
    """
    Transform audio data, combine it with final latents from a projection file,
    and feed the result into a network for synthesis. Optionally overlay parts of the target
    video inside of the projection file onto the output video.

    Note: Audio data will be scaled to the duration of the projection file.

    \f
    :param wav: See click help.
    :param output_path: See click help.
    :param networks_directory: See click help.
    :param network_path: See click help.
    :param networks_json: See click help.
    :param frames_to_visualize: See click help.
    :param output_fps: See click help.
    :param output_side_length: See click help.
    :param debug_path: See click help.
    :param debug_window: See click help.
    :param alpha: See click help.
    :param fft_roll_enabled: See click help.
    :param fft_amplitude_range: See click help.
    :param run_config: See click help.
    :param projection_file_path: See click help.
    :param blend_depth: See click help.
    :param complexity_change_rolling_sum_window: See click help.
    :param complexity_change_threshold: See click help.
    :param phash_distance: See click help.
    :param bbox_distance: See click help.
    :param track_length: See click help.
    :return: None
    """

    network_paths = parse_network_paths(
        networks_directory=networks_directory, networks=network_path, networks_json=networks_json
    )

    with MultiNetwork(network_paths=network_paths) as multi_networks:
        _projection_file_blend_api(
            wav=wav,
            output_path=output_path,
            multi_networks=multi_networks,
            frames_to_visualize=frames_to_visualize,
            output_fps=output_fps,
            output_side_length=output_side_length,
            debug_path=debug_path,
            debug_window=debug_window,
            alpha=alpha,
            fft_roll_enabled=fft_roll_enabled,
            fft_amplitude_range=fft_amplitude_range,
            run_config=run_config,
            projection_file_path=projection_file_path,
            blend_depth=blend_depth,
            complexity_change_rolling_sum_window=complexity_change_rolling_sum_window,
            complexity_change_threshold=complexity_change_threshold,
            phash_distance=phash_distance,
            bbox_distance=bbox_distance,
            track_length=track_length,
        )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
