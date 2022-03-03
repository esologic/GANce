"""Uses pathlib to make referencing test assets by path easier."""

import os
from pathlib import Path

_ASSETS_DIRECTORY = Path(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIRECTORY_PATH = str(_ASSETS_DIRECTORY)

FACE_IMAGE_PATH = _ASSETS_DIRECTORY.joinpath("face_image.jpeg")
NO_FACE_IMAGE_PATH = _ASSETS_DIRECTORY.joinpath("no_face_image.jpeg")

SAMPLE_BATCH_1_NETWORK_PATH = _ASSETS_DIRECTORY.joinpath("sample_batch_1_network.pkl")
SAMPLE_BATCH_2_NETWORK_PATH = _ASSETS_DIRECTORY.joinpath("sample_batch_2_network.pkl")

BATCH_2_IMAGE_1_PATH = _ASSETS_DIRECTORY.joinpath("batch_2_sample_1_image.jpeg")
BATCH_2_IMAGE_2_PATH = _ASSETS_DIRECTORY.joinpath("batch_2_sample_2_image.jpeg")

SAMPLE_FACE_VIDEO_PATH = _ASSETS_DIRECTORY.joinpath("devon_face_sample.mp4")
SAMPLE_FACE_VIDEO_SHORT_PATH = _ASSETS_DIRECTORY.joinpath("devon_face_sample_short.mp4")
SAMPLE_FACE_VIDEO_EXPECTED_WIDTH_HEIGHT = (1024, 1024)
SAMPLE_FACE_VIDEO_EXPECTED_FPS = 30.0
SAMPLE_FACE_VIDEO_EXPECTED_FRAMES_COUNT = 520

SAMPLE_PROJECTION_FILE_PATH = _ASSETS_DIRECTORY.joinpath("sample_projection_file.hdf5")
SAMPLE_SYNTHESIS_FILE_PATH = _ASSETS_DIRECTORY.joinpath("sample_synthesis_file.json")

WAV_CLAPS_PATH = _ASSETS_DIRECTORY.joinpath("claps.wav")
