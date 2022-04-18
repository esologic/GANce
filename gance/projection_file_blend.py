"""
Transform audio data, combine it with final latents from a projection file,
and feed the result into a network for synthesis. Optionally overlay parts of the target
video inside of the projection file onto the output video.
"""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, cast

import cv2
import more_itertools
import numpy as np
import pandas as pd
from lz.transposition import transpose

from gance import divisor, overlay
from gance.data_into_network_visualization import visualize_vector_reduction
from gance.data_into_network_visualization.network_visualization import vector_synthesis
from gance.data_into_network_visualization.visualization_common import DataLabel, ResultLayers
from gance.data_into_network_visualization.visualization_inputs import alpha_blend_projection_file
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.image_sources import image_sources_common, video_common
from gance.image_sources.image_sources_common import ImageResolution
from gance.iterator_on_disk import HDF5_SERIALIZER, iterator_on_disk
from gance.logger_common import LOGGER
from gance.network_interface.network_functions import MultiNetwork
from gance.projection import projection_file_reader
from gance.vector_sources import music, vector_reduction
from gance.vector_sources.vector_sources_common import underlying_length
from gance.vector_sources.vector_types import ConcatenatedVectors


def _create_iterators_on_disk(
    iterators: Tuple[ImageSourceType, ...], num_copies: int
) -> Tuple[Iterator[ImageSourceType], ...]:
    """
    Helper function, canonical invocation for this file.
    :param iterators: To duplicate on disk.
    :param num_copies: Number of on-disk copies of each iterator to make.
    :return: A tuple of the on-disk iterators.
    """

    return tuple(
        iter(
            iterator_on_disk(
                iterator=iterator,
                copies=num_copies,
                serializer=HDF5_SERIALIZER,
            )
        )
        for iterator in iterators
    )


def horizontal_concat_images(images: Iterator[RGBInt8ImageType]) -> RGBInt8ImageType:
    """
    Helper function. Adds logging.
    :param images: To concatenate.
    :return: Concatenated image.
    """
    images_as_list = list(images)
    LOGGER.debug(
        f"Horizontally concatenating {len(images_as_list)} images, "
        f"sizes: {[image_sources_common.image_resolution(image) for image in images_as_list]}"
    )
    output: RGBInt8ImageType = cv2.hconcat(images_as_list)
    return output


def scale_square_source(
    source: ImageSourceType, output_side_length: int, frame_multiplier: int
) -> ImageSourceType:
    """
    Scale the resolution and number of frames in a given source.
    :param source: To scale.
    :param output_side_length: Square frames will be resized to this side length.
    :param frame_multiplier: Every frame will be duplicated this many times.
    :return: Scaled source.
    """
    return cast(
        ImageSourceType,
        more_itertools.repeat_each(
            video_common.resize_source(
                source, ImageResolution(output_side_length, output_side_length)
            ),
            frame_multiplier,
        ),
    )


def projection_file_blend_api(  # pylint: disable=too-many-arguments,too-many-locals
    wav: List[str],
    output_path: str,
    network_paths: List[Path],
    frames_to_visualize: Optional[int],
    output_fps: float,
    output_side_length: int,
    debug_path: Optional[str],
    debug_window: Optional[int],
    alpha: float,
    fft_roll_enabled: bool,
    fft_amplitude_range: Tuple[int, int],
    projection_file_path: str,
    blend_depth: int,
    complexity_change_rolling_sum_window: Optional[int],
    complexity_change_threshold: Optional[int],
    phash_distance: Optional[int],
    bbox_distance: Optional[float],
    track_length: Optional[int],
) -> None:
    """
    API function to omit input arguments from the CLI, see the main CLI function for more complete
    docs.

    \f
    :param wav: See click help.
    :param output_path: See click help.
    :param network_paths: Paths to the network pickle files that should be used in the run.
    :param frames_to_visualize: See click help.
    :param output_fps: See click help.
    :param output_side_length: See click help.
    :param debug_path: See click help.
    :param debug_window: See click help.
    :param alpha: See click help.
    :param fft_roll_enabled: See click help.
    :param fft_amplitude_range: See click help.
    :param projection_file_path: See click help.
    :param blend_depth: See click help.
    :param complexity_change_rolling_sum_window: The number of frames to window the music
    complexity computation to.
    :param complexity_change_threshold: If complexity is under this value, an overlay computation
    is enabled.
    :param phash_distance: See click help.
    :param bbox_distance: See click help.
    :param track_length: See click help.
    :return: None
    """

    create_debug_visualization = debug_path is not None

    audio_paths = list(map(Path, wav))

    overlay_enabled = all(
        param is not None for param in (phash_distance, bbox_distance, track_length)
    )

    overlay_music_mask_enabled = all(
        param is not None
        for param in (complexity_change_rolling_sum_window, complexity_change_threshold)
    )

    if overlay_music_mask_enabled and not overlay_enabled:
        raise ValueError("Overlay music mask without overlay being enabled is not supported!")

    with MultiNetwork(
        network_paths=network_paths
    ) as multi_networks, projection_file_reader.load_projection_file(
        Path(projection_file_path)
    ) as reader:

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

        num_output_frames = int(frame_multiplier * final_latents_in_file)

        time_series_audio_vectors = cast(
            ConcatenatedVectors,
            music.read_wavs_scale_for_video(
                wavs=audio_paths,
                vector_length=multi_networks.expected_vector_length,
                target_num_vectors=num_output_frames,
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

        foreground_iterators, background_iterators = _create_iterators_on_disk(
            iterators=(
                # Foreground iterator, the projection targets.
                scale_square_source(
                    source=reader.target_images,
                    output_side_length=output_side_length,
                    frame_multiplier=frame_multiplier,
                ),
                # Background iterator, the network outputs.
                synthesis_output.synthesized_images,
            ),
            num_copies=sum([overlay_enabled]),
        )

        music_complexity_overlay_mask: Optional[ResultLayers] = (
            vector_reduction.rolling_sum_results_layers(
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
            if overlay_music_mask_enabled
            else None
        )

        if overlay_enabled:

            # Don't want to skip any frames if we're not using audio as an input here.
            skip_mask: List[bool] = (
                list(
                    pd.Series(music_complexity_overlay_mask.result.data).fillna(np.inf)
                    > complexity_change_threshold
                )
                if overlay_music_mask_enabled
                else [False] * num_output_frames
            )

            overlay_results = overlay.overlay_eye_tracking.compute_eye_tracking_overlay(
                foreground_images=next(foreground_iterators),
                background_images=next(background_iterators),
                min_phash_distance=phash_distance,
                min_bbox_distance=bbox_distance,
                skip_mask=skip_mask,
            )

            logging.info(
                "Starting to compute mask to filter out short sequences of overlay frames."
            )

            boxes_list = list(overlay_results.bbox_lists)

            long_tracks_mask = vector_reduction.track_length_filter(
                bool_tracks=(
                    ~pd.Series(skip_mask) & pd.Series((box is not None for box in boxes_list))
                ),
                track_length=track_length,
            )

            final_frames_and_foregrounds: Iterator[Tuple[RGBInt8ImageType, RGBInt8ImageType]] = (
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

            blended_output, foregrounds = transpose(final_frames_and_foregrounds)
        else:
            blended_output = next(background_iterators)
            foregrounds = None

        blended_output = video_common.write_source_to_disk_forward(
            source=blended_output,
            video_path=Path(output_path),
            video_fps=output_fps,
            audio_paths=audio_paths,
            high_quality=True,
        )

        if create_debug_visualization:
            overlay_visualization = (
                overlay.overlay_visualization.visualize_overlay_computation(
                    overlay=overlay_results.contexts,
                    frames_per_context=debug_window,
                    video_square_side_length=output_side_length,
                    horizontal_lines=overlay.overlay_visualization.VisualizeOverlayThresholds(
                        phash_line=phash_distance, bbox_distance_line=bbox_distance
                    ),
                )
                if overlay_enabled
                else None
            )

            video_common.write_source_to_disk_consume(
                source=(
                    horizontal_concat_images(images)
                    for images in zip(
                        *filter(
                            lambda optional_iterable: optional_iterable is not None,
                            [
                                blended_output,
                                foregrounds,
                                scale_square_source(
                                    source=reader.final_images,
                                    output_side_length=output_side_length,
                                    frame_multiplier=frame_multiplier,
                                ),
                                synthesis_output.visualization_images,
                                overlay_visualization,
                                visualize_vector_reduction.visualize_result_layers(
                                    result_layers=music_complexity_overlay_mask,
                                    frames_per_context=debug_window,
                                    video_height=output_side_length,
                                    title="Overlay binary mask",
                                    horizontal_line=complexity_change_threshold,
                                )
                                if music_complexity_overlay_mask is not None
                                else None,
                            ],
                        )
                    )
                ),
                video_path=Path(debug_path),
                video_fps=output_fps,
                audio_paths=audio_paths,
                high_quality=True,
            )
        else:
            more_itertools.consume(blended_output)
