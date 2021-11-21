"""Uses pathlib to make referencing assets by path easier."""

import os
from pathlib import Path

_ASSETS_DIRECTORY = Path(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIRECTORY_PATH = str(_ASSETS_DIRECTORY)

WAV_CLAPS_PATH = _ASSETS_DIRECTORY.joinpath("claps.wav")

OUTPUT_DIRECTORY = _ASSETS_DIRECTORY.joinpath("output")
