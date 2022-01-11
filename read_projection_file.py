"""
CLI to do common things with a projection file.
"""

import shutil
import tempfile
from pathlib import Path
import face_detection
from typing import List, Optional


import itertools
import click
from skimage import data
from skimage.feature import Cascade
from gance import cli_common
from gance.projection import projection_visualization
from gance.video_common import add_wavs_to_video
from pathlib import Path
from typing import List, Optional
from PIL import Image

import numpy as np
from cv2 import cv2

from gance.data_into_model_visualization.vectors_to_image import vector_visualizer

from gance.logger_common import LOGGER
from gance.projection.projection_file_reader import (
    final_latents_matrices_label,
    load_projection_file,
)
from gance.vector_sources.vector_sources_common import sub_vectors
from gance.vector_sources.vector_types import SingleMatrix
from gance.video_common import create_video_writer


@click.group()
def cli() -> None:
    """
    Use one of the commands below to process a projection file.

    \f

    :return: None
    """


@cli.command()
@cli_common.single_projection_file_path
@cli_common.video_path
@cli_common.audio_paths
@cli_common.video_height
def visualize_final_latents(  # pylint: disable=too-many-arguments,too-many-locals
    projection_file: str,
    video_path: str,
    audio_path: Optional[List[str]],
    video_height: Optional[int],
) -> None:
    """
    Create a video that, side by side, displays:

    * Final Latents as a graph.
    * Target image.
    * Final latents image.

    \f

    :param projection_file: See click help or called function docs.
    :param video_path: See click help or called function docs.
    :param audio_path: Optional path to audio file.
    :param video_height: See click help or called function docs.
    :return: None
    """

    output_video_path = Path(video_path)

    with tempfile.NamedTemporaryFile(suffix=output_video_path.suffix) as f:

        tmp_video_path = Path(f.name)

        projection_visualization.visualize_final_latents(
            projection_file_path=Path(projection_file),
            output_video_path=tmp_video_path,
            video_height=video_height,
        )

        if audio_path:
            while not tmp_video_path.exists():
                pass

            add_wavs_to_video(
                video_path=tmp_video_path,
                audio_paths=[Path(path) for path in audio_path],
                output_path=output_video_path,
            )
        else:
            shutil.move(src=str(tmp_video_path), dst=str(output_video_path))


@cli.command()
@cli_common.single_projection_file_path
@cli_common.video_path
@cli_common.audio_paths
@cli_common.video_height
def overlay(  # pylint: disable=too-many-arguments,too-many-locals
    projection_file: str,
    video_path: str,
    audio_path: Optional[List[str]],
    video_height: Optional[int],
) -> None:
    """

    :param projection_file:
    :param video_path:
    :param audio_path:
    :param video_height:
    :return:
    """

    output_video_path = Path(video_path)

    face_detector = Cascade(data.lbp_frontal_face_cascade_filename())

    with load_projection_file(Path(projection_file)) as reader:

        matrices_label = final_latents_matrices_label(reader)

        video = create_video_writer(
            video_path=output_video_path,
            num_squares=3,
            video_fps=reader.projection_attributes.projection_fps,
            video_height=video_height,
        )

        num_matrices = len(sub_vectors(
            data=matrices_label.data, vector_length=matrices_label.vector_length
        ))

        for index, (target, final_image) in itertools.islice(enumerate(
            zip(reader.target_images, reader.final_images)
        ), 200):

            new_image = Image.fromarray(np.array(final_image))
            target_image = Image.fromarray(target)

            boxes = face_detector.detect_multi_scale(
                    img=target,
                    scale_factor=1.2,
                    step_ratio=1,
                    min_size=(100, 100),
                    max_size=(1000, 1000),
                )

            for box in boxes:
                x, y, w, h = (box['r'],  box['c'],  box['width'], box['height'])
                crop = (x, y, x + w, y + h)
                cropped_box = target_image.crop(crop)
                new_image.paste(cropped_box, crop)

            frame = cv2.hconcat([np.asarray(new_image), target, final_image])

            video.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

            LOGGER.info(f"Wrote frame: {output_video_path}, frame: {index + 1}/{num_matrices}")

        video.release()


if __name__ == "__main__":
    cli()
