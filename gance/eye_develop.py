"""
Functions to feed vectors into a model and record the output.
Also tools to visualize these vectors against the model outputs.
"""
import logging
import time
from typing import Iterator, List, Tuple

import more_itertools
import numpy as np
import pandas as pd
from cv2 import cv2
from lz.transposition import transpose

from gance import divisor, overlay
from gance.assets import NOVA_PATH, OUTPUT_DIRECTORY, PRODUCTION_MODEL_PATH, PROJECTION_FILE_PATH
from gance.data_into_model_visualization import visualize_vector_reduction
from gance.data_into_model_visualization.model_visualization import viz_model_ins_outs
from gance.data_into_model_visualization.visualization_inputs import alpha_blend_projection_file
from gance.gance_types import RGBInt8ImageType
from gance.image_sources import video_common
from gance.iterator_on_disk import HDF5_SERIALIZER, iterator_on_disk
from gance.model_interface.model_functions import MultiModel
from gance.projection import projection_file_reader
from gance.vector_sources import music, vector_reduction
from gance.vector_sources.vector_reduction import DataLabel, ResultLayers

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


def main() -> None:  # pylint: disable=too-many-locals
    """

    :return:
    """

    frames_to_visualize = None
    video_fps = 30
    context_windows_length = 200
    video_square_side_length = 1024
    full_context = True

    with MultiModel(model_paths=[PRODUCTION_MODEL_PATH]) as multi_models:

        with projection_file_reader.load_projection_file(PROJECTION_FILE_PATH) as reader:

            time_series_audio_vectors = music.read_wav_scale_for_video(
                wav=NOVA_PATH,
                vector_length=multi_models.expected_vector_length,
                frames_per_second=video_fps,
                cache_path=OUTPUT_DIRECTORY.joinpath("wav_cache.p"),
            )

            overlay_mask = vector_reduction.rolling_sum_results_layers(
                vector_reduction.absolute_value_results_layers(
                    results_layers=ResultLayers(
                        result=DataLabel(
                            data=vector_reduction.derive_results_layers(
                                vector_reduction.reduce_vector_gzip_compression_rolling_average(
                                    time_series_audio_vectors=time_series_audio_vectors.wav_data,
                                    vector_length=multi_models.expected_vector_length,
                                ),
                                order=1,
                            ).result.data,
                            label="Gzipped audio, smoothed, averaged, 1st order derivation.",
                        ),
                    ),
                ),
                window_length=video_fps * 1,
            )

            skip_mask: List[bool] = list(pd.Series(overlay_mask.result.data).fillna(np.inf) > 100)

            # Variable here is to avoid long line.
            final_latents = projection_file_reader.final_latents_matrices_label(reader)

            model_output = viz_model_ins_outs(
                models=multi_models,
                data=alpha_blend_projection_file(
                    final_latents_matrices_label=final_latents,
                    alpha=0.25,
                    fft_roll_enabled=True,
                    fft_amplitude_range=(-1, 1),
                    blend_depth=10,
                    time_series_audio_vectors=time_series_audio_vectors.wav_data,
                    vector_length=multi_models.expected_vector_length,
                    model_indices=multi_models.model_indices,
                ),
                default_vector_length=multi_models.expected_vector_length,
                enable_3d=False,
                enable_2d=full_context,
                frames_to_visualize=frames_to_visualize,
                model_index_window_width=context_windows_length,
            )

            foreground_iterators = iter(
                iterator_on_disk(
                    iterator=more_itertools.repeat_each(
                        reader.target_images,
                        divisor.divide_no_remainder(
                            numerator=video_fps,
                            denominator=reader.projection_attributes.projection_fps,
                        ),
                    ),
                    copies=1,
                    serializer=HDF5_SERIALIZER,
                )
            )

            background_iterators = iter(
                iterator_on_disk(
                    iterator=model_output.model_images,
                    copies=1,
                    serializer=HDF5_SERIALIZER,
                )
            )

            overlay_results = overlay.compute_eye_tracking_overlay(
                foreground_images=next(foreground_iterators),
                background_images=next(background_iterators),
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
                track_length=20,
            )

            final_frames: Iterator[Tuple[RGBInt8ImageType, RGBInt8ImageType, RGBInt8ImageType]] = (
                (
                    overlay.write_boxes_onto_image(
                        foreground_image=foreground,
                        background_image=background,
                        bounding_boxes=bounding_boxes,
                    )
                    if in_long_track
                    else background,
                    foreground,
                    background,
                )
                for (bounding_boxes, foreground, background, in_long_track) in zip(
                    boxes_list,
                    next(foreground_iterators),
                    next(background_iterators),
                    long_tracks_mask,
                )
            )

            finals, foregrounds, backgrounds = transpose(final_frames)

            current_time = int(time.time())

            finals = video_common.write_source_to_disk_forward(
                source=finals,
                video_path=OUTPUT_DIRECTORY.joinpath(f"{current_time}_hero_only.mp4"),
                video_fps=video_fps,
                audio_path=NOVA_PATH,
            )

            if full_context:

                overlay_visualization = overlay.visualize_overlay_computation(
                    overlay=overlay_results.contexts,
                    frames_per_context=context_windows_length,
                    video_square_side_length=video_square_side_length,
                )

                music_overlay_mask_visualization = (
                    visualize_vector_reduction.visualize_result_layers(
                        result_layers=overlay_mask,
                        frames_per_context=context_windows_length,
                        video_height=video_square_side_length,
                        title="Overlay binary mask",
                        horizontal_line=75,
                    )
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
                            backgrounds,
                            overlay_visualization,
                            model_output.visualization_images,
                            music_overlay_mask_visualization,
                        )
                    ),
                    video_path=OUTPUT_DIRECTORY.joinpath(f"{current_time}_full_context.mp4"),
                    video_fps=video_fps,
                    audio_path=NOVA_PATH,
                )
            else:
                more_itertools.consume(finals)


if __name__ == "__main__":
    main()
