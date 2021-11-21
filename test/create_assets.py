"""
Run this script to re-create test some of the test assets from scratch.
Requires a GPU.
"""

from test.assets import (
    SAMPLE_BATCH_2_MODEL_PATH,
    SAMPLE_FACE_VIDEO_SHORT_PATH,
    SAMPLE_PROJECTION_FILE_PATH,
)

from gance.logger_common import LOGGER
from gance.projection.projector_file_writer import project_video_to_file

if __name__ == "__main__":

    LOGGER.info(f"Generating {SAMPLE_PROJECTION_FILE_PATH}")

    project_video_to_file(
        path_to_model=SAMPLE_BATCH_2_MODEL_PATH,
        path_to_video=SAMPLE_FACE_VIDEO_SHORT_PATH,
        projection_file_path=SAMPLE_PROJECTION_FILE_PATH,
    )
