import os
import pathlib

from setuptools import setup

REQUIREMENTS = ["numpy", "torch", "SimpleITK", "pynrrd", "medpy",
                "raster_geometry", "requests", "typer", "tensorboard",
                "pandas", "scikit-image"]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
VERSION = "0.102"
DESCRIPTION = "3D UNet for CT segmentation with PyTorch"

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name="ctunet",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=["Franco Matzkin"],
    author_email="fmatzkin@sinc.unl.edu.ar",
    url="https://github.com/vfmatzkin/ctunet",
    license="MIT",
    keywords=["ctunet"],
    packages=["ctunet", "ctunet.pytorch"],
    install_requires=REQUIREMENTS,
    include_package_data=True,
    entry_points={"console_scripts": ["ctunet=ctunet.pytorch.train_test:cli"]},
)
