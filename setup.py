import os
import pathlib

from setuptools import setup

REQUIREMENTS = ["numpy", "torch==1.5.0", "SimpleITK", "pynrrd", "medpy",
                "raster_geometry", "requests", "typer", "tensorboard",
                "pandas", "scikit-image", "antspyx"]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
VERSION = "0.15"
DESCRIPTION = "A set of tools for preproccessing and performing brain " \
              "segmentation and skull reconstruction on head CT images"

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name="headctools",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=["Franco Matzkin"],
    author_email="fmatzkin@sinc.unl.edu.ar",
    url="https://gitlab.com/matzkin/headctools",
    license="MIT",
    keywords=["headctools, deep-brain-extractor"],
    packages=["headctools", "headctools.pytorch", "headctools.tools",
              "headctools.assets.download"],
    entry_points={"console_scripts": ["headctools=headctools.cli:main"]},
    install_requires=REQUIREMENTS,
    include_package_data=True,
)
