"""Uses pathlib to make referencing assets by path easier."""

import os
from pathlib import Path

_ASSETS_DIRECTORY = Path(os.path.dirname(os.path.abspath(__file__)))

ASSETS_DIRECTORY_PATH = str(_ASSETS_DIRECTORY)

_AUDIO_DIRECTORY = _ASSETS_DIRECTORY.joinpath("audio")

WAV_CLAPS_PATH = _AUDIO_DIRECTORY.joinpath("claps.wav")

NOVA_SNIPPET_PATH = _AUDIO_DIRECTORY.joinpath("nova_prod_snippet.wav")
NOVA_PATH = _AUDIO_DIRECTORY.joinpath("nova_prod.wav")

OUTPUT_DIRECTORY = _ASSETS_DIRECTORY.joinpath("output")

_NETWORKS_DIRECTORY = _ASSETS_DIRECTORY.joinpath("networks")

PRODUCTION_NETWORK_PATH = _NETWORKS_DIRECTORY.joinpath("production_network.pkl")

TRAINING_SAMPLE_NETWORKS_DIRECTORY = _NETWORKS_DIRECTORY.joinpath("training_sample")
TRAINING_COMPLETE_NETWORKS_DIRECTORY = _NETWORKS_DIRECTORY.joinpath("training_complete")

TRAINING_SAMPLE_NETWORKS_DIRECTORY.joinpath("1.pkl")

EARLY_TRAINING_NETWORK_PATH = TRAINING_SAMPLE_NETWORKS_DIRECTORY.joinpath("1.pkl")
MID_TRAINING_NETWORK_PATH = TRAINING_SAMPLE_NETWORKS_DIRECTORY.joinpath("3.pkl")

PROJECTION_FILE_PATH = _ASSETS_DIRECTORY.joinpath("projection_files/resumed_prod_nova_3-1.hdf5")
