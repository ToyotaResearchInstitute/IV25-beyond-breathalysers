#!/usr/bin/env python3

from setuptools import find_packages, setup

__version__ = "0.1.0"

setup(
    name="IV25",
    version=__version__,
    packages=find_packages(exclude=["doc*"]),
    author="simon.stent@tri.global",
)