"""
Happy path tests
"""

from test.assets import BATCH_2_IMAGE_1_PATH, BATCH_2_IMAGE_2_PATH

from gance import overlay
from gance.image_sources import still_image_common


def test_write_boxes_onto_image() -> None:
    """
    This function should run without error. Verified by looking at result image.
    :return: None
    """

    overlay.write_boxes_onto_image(
        foreground_image=still_image_common.read_image(image_path=BATCH_2_IMAGE_1_PATH),
        background_image=still_image_common.read_image(image_path=BATCH_2_IMAGE_2_PATH),
        bounding_boxes=[overlay.BoundingBox(x=0, y=0, width=100, height=100)],
    )
