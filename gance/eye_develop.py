"""
Functions to feed vectors into a model and record the output.
Also tools to visualize these vectors against the model outputs.
"""

import itertools
import logging
import time
from typing import cast

import more_itertools
from cv2 import cv2

from gance import divisor, overlay
from gance.assets import NOVA_PATH, OUTPUT_DIRECTORY, PRODUCTION_MODEL_PATH, PROJECTION_FILE_PATH
from gance.data_into_model_visualization.model_visualization import viz_model_ins_outs
from gance.data_into_model_visualization.visualization_inputs import alpha_blend_projection_file
from gance.gance_types import ImageSourceType
from gance.image_sources import video_common
from gance.model_interface.model_functions import MultiModel
from gance.projection import projection_file_reader
from gance.vector_sources import music

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)


if __name__ == "__main__":

    frames_to_visualize = None
    video_fps = 60

    with MultiModel(model_paths=[PRODUCTION_MODEL_PATH]) as multi_models:

        with projection_file_reader.load_projection_file(PROJECTION_FILE_PATH) as reader:

            wav = music.read_wav_file(NOVA_PATH)

            time_series_audio_vectors = music.read_wav_scale_for_video(
                wav=wav,
                vector_length=multi_models.expected_vector_length,
                frames_per_second=video_fps,
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

            final_latents = projection_file_reader.final_latents_matrices_label(reader)

            model_output = viz_model_ins_outs(
                models=multi_models,
                data=alpha_blend_projection_file(
                    final_latents_matrices_label=final_latents,
                    alpha=0.5,
                    fft_roll_enabled=True,
                    fft_amplitude_range=(-5, 5),
                    blend_depth=10,
                    time_series_audio_vectors=time_series_audio_vectors.wav_data,
                    vector_length=multi_models.expected_vector_length,
                    model_indices=multi_models.model_indices,
                ),
                default_vector_length=multi_models.expected_vector_length,
                enable_3d=False,
                enable_2d=True,
                frames_to_visualize=frames_to_visualize,
                model_index_window_width=100,
            )

            overlay_results = overlay.compute_eye_tracking_overlay(
                foreground_images=itertools.islice(targets, frames_to_visualize),
                background_images=model_output.model_images,
            )

            overlay_visualization = overlay.visualize_overlay_computation(
                overlay=overlay_results.contexts,
                frames_per_context=100,
                video_square_side_length=1024,
            )

            frames = (
                cv2.hconcat(
                    [
                        overlay.apply_mask(
                            foreground_image=foreground,
                            background_image=background,
                            mask=mask,
                        ),
                        background,
                        foreground,
                        overlay_visualization_frame,
                        visualization_image,
                    ]
                )
                for (
                    mask,
                    foreground,
                    background,
                    overlay_visualization_frame,
                    visualization_image,
                ) in zip(
                    overlay_results.masks,
                    overlay_results.foregrounds,
                    overlay_results.backgrounds,
                    overlay_visualization,
                    model_output.visualization_images,
                )
            )

            video_path = OUTPUT_DIRECTORY.joinpath(f"{int(time.time())}_overlay_test.mp4")

            video_common.write_source_to_disk(
                source=frames,
                video_path=video_path,
                video_fps=video_fps,
            )

            video_common.add_wav_to_video(
                video_path=video_path,
                audio_path=NOVA_PATH,
                output_path=OUTPUT_DIRECTORY.joinpath(f"{int(time.time())}_overlay_test_audio.mp4"),
            )
