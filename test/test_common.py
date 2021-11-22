"""
Common functions used across tests.
"""

from pathlib import Path
from time import sleep


def wait_get_size(file: Path) -> int:
    """
    Wait for file to appear on disk then return it's size in bytes.
    :param file: Path to file.
    :return: File size in bytes
    """
    for _ in range(5):
        if file.exists():
            break
        sleep(1)
    else:
        raise FileNotFoundError(f"Waiting for {file} to appear but it didn't")

    return file.stat().st_size
