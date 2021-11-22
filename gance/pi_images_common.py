"""
Common code for dealing with images captured by the Raspberry Pi
TODO: We could import this from the other project: https://github.com/esologic/pitraiture
"""

from datetime import datetime

IMAGE_EXTENSION = "jpeg"
IMAGE_TIMESTAMP_FORMAT = "%m-%d-%Y_%H-%M-%S-%f"


def create_image_filename(dataset_name: str, capture_time: datetime) -> str:
    """
    Helper function to create the image filename given the dataset name and the time
    it was captured.
    :param dataset_name: The name of the dataset the resulting image will belong to.
    :param capture_time: The time the image was captured as a datetime.
    :return: The image name, ex:
    `april_27_cottage_session_1_04-28-2021_11-50-12-752379.jpeg`
    """
    return f"{dataset_name}_{capture_time.strftime(IMAGE_TIMESTAMP_FORMAT)}.{IMAGE_EXTENSION}"
