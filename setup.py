"""Use this file to install gance as a module"""

from distutils.core import setup
from setuptools import find_packages
from typing import List


def prod_dependencies() -> List[str]:
    """
    Pull the dependencies from the requirements dir
    :return: Each of the newlines, strings of the dependencies
    """
    with open("./requirements/prod.txt", "r") as file:
        return file.read().splitlines()


setup(
    name="gance",
    version="0.18.0",
    description="Maps music and video into the latent space of StyleGAN models.",
    author="Devon Bray",
    author_email="dev@esologic.com",
    packages=find_packages(),
    install_requires=prod_dependencies(),
)
