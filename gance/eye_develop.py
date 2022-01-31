"""
Functions to feed vectors into a model and record the output.
Also tools to visualize these vectors against the model outputs.
"""

import itertools
import logging
import time
from typing import Iterator, List, Tuple, cast

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
from gance.gance_types import ImageSourceType, RGBInt8ImageType
from gance.image_sources import video_common
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

            wav = music.read_wav_file(NOVA_PATH)

            time_series_audio_vectors = music.read_wav_scale_for_video(
                wav=wav,
                vector_length=multi_models.expected_vector_length,
                frames_per_second=video_fps,
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

            for_mask = pd.Series(overlay_mask.result.data).fillna(np.inf)

            skip_mask: List[bool] = list(for_mask > 75)

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

            targets = cast(
                ImageSourceType,
                more_itertools.repeat_each(
                    reader.target_images,
                    divisor.divide_no_remainder(
                        numerator=video_fps, denominator=reader.projection_attributes.projection_fps
                    ),
                ),
            )

            overlay_results = overlay.compute_eye_tracking_overlay(
                foreground_images=itertools.islice(targets, frames_to_visualize),
                background_images=model_output.model_images,
                skip_mask=skip_mask,
            )

            overlay_visualization = overlay.visualize_overlay_computation(
                overlay=overlay_results.contexts,
                frames_per_context=context_windows_length,
                video_square_side_length=video_square_side_length,
            )

            music_overlay_mask_visualization = visualize_vector_reduction.visualize_result_layers(
                result_layers=overlay_mask,
                frames_per_context=context_windows_length,
                video_height=video_square_side_length,
                title="Overlay binary mask",
                horizontal_line=75,
            )

            final_frames: Iterator[Tuple[RGBInt8ImageType, RGBInt8ImageType, RGBInt8ImageType]] = (
                (
                    overlay.apply_mask(
                        foreground_image=foreground,
                        background_image=background,
                        mask=mask,
                    ),
                    foreground,
                    background,
                )
                for (mask, foreground, background) in zip(
                    overlay_results.masks,
                    overlay_results.foregrounds,
                    overlay_results.backgrounds,
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


if __name__ == "__main__":
    main()
