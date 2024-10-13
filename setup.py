import sys

if sys.version_info < (3, 6):
    sys.exit("garfield requires Python >= 3.6")

from setuptools import setup, find_packages
from pathlib import Path

version = {}
with open("Garfield/_version.py") as fp:
    exec(fp.read(), version)

setup(
    name="Garfield",
    version=version["__version__"],
    author="Weige Zhou",
    author_email="zhouwg1314@gmail.com",
    license="BSD",
    description="Garfield: Graph-based Contrastive Learning enable Fast Single-Cell Embedding",
    long_description=Path("README.md").read_text("utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/zhou-1314/Garfield",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        x.strip() for x in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    include_package_data=True,
    package_data={"Garfield": ["data/gene_anno/*.bed"], "": ["*.txt"]},
)
