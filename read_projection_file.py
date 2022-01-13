"""
CLI to do common things with a projection file.
"""

import shutil
import tempfile
from pathlib import Path
import face_detection
from typing import List, Optional
import face_recognition

from gance.assets import PROJECTION_FILE_PATH
import itertools
import click
from skimage import data
from skimage.feature import Cascade
from gance import cli_common
from gance.projection import projection_visualization
from gance.video_common import add_wavs_to_video
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageDraw

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
def fuck():
    pass

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

        for index, (target, final) in itertools.islice(enumerate(
            zip(reader.target_images, reader.final_images)
        ), 200):

            target_image = Image.fromarray(target)
            final_image = Image.fromarray(final)

            landmarks = face_recognition.face_landmarks(face_image=target)

            mask = Image.new('L', (target_image.size[1], target_image.size[0]), 0)

            for landmark in landmarks:
                draw = ImageDraw.Draw(mask)
                draw.polygon(landmark["left_eye"], outline=255, fill=255)
                draw.polygon(landmark["right_eye"], outline=255, fill=255)

            new_image = Image.composite(target_image, final_image, mask)

            frame = cv2.hconcat([np.asarray(new_image), target, final])

            video.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB))

            LOGGER.info(f"Wrote frame: {output_video_path}, frame: {index + 1}/{num_matrices}")

        video.release()


if __name__ == "__main__":

    overlay(
        projection_file=PROJECTION_FILE_PATH,
        video_path="./output.mp4",
        video_height=1024,
        audio_path=None,
    )

    # cli()
