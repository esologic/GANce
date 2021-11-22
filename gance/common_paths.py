"""
Common paths
"""

from pathlib import Path

_HDD_PATH = Path("/home/devon/outside_hdd")

_PI_HHD_PATH = Path("/home/pi/hdd")
DEFAULT_PI_DATASET_ROOT = _PI_HHD_PATH.joinpath("datasets")

# Any file that gets moved to this location will get picked up by ownCloud and uploaded.
DEFAULT_PI_UPLOAD_LOCATION = _PI_HHD_PATH.joinpath("ownCloud").joinpath("pi_dataset_dropbox")

FACE_IMAGE_EXTENSION = ".png"

_DEVON_IMAGES_ROOT = _HDD_PATH.joinpath("devon_face_images")

DEVON_ALL_CANDIDATE_IMAGES_PATH = _DEVON_IMAGES_ROOT.joinpath("all")
DEVON_FACE_IMAGES_PATH = _DEVON_IMAGES_ROOT.joinpath("images_with_faces")
DEVON_CROPPED_FACE_IMAGES_PATH = _DEVON_IMAGES_ROOT.joinpath("cropped_face_images")
DEVON_SCALED_CROPPED_FACE_IMAGES_PATH = _DEVON_IMAGES_ROOT.joinpath("1024_1024_cropped_face_images")
TWENTY_GB_OF_DEVON_SCALED_CROPPED_FACE_IMAGES_PATH = _DEVON_IMAGES_ROOT.joinpath(
    "20gb_of_1024_1024_cropped_face_images"
)
