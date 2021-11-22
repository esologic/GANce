"""
Lib function to quickly hash a file on disk.
"""

import hashlib
from pathlib import Path


def hash_file(path_to_file: Path) -> str:
    """
    Hash a file.
    Per: https://stackoverflow.com/a/59056837
    :param path_to_file: File to hash.
    :return: Hex digest of MD5 hash of contents of file.
    """

    with open(str(path_to_file), "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

    return file_hash.hexdigest()
