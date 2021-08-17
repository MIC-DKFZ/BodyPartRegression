import os
from setuptools import setup, find_packages
from setuptools.command.install import install

# read readme
with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="bpreg",
    version="1.0.2", # 1.0.0
    packages=find_packages(),
    include_package_data=True,
    test_suite="unittest",
    install_requires=[
        "pytorch_lightning==1.2.10", 
        "nibabel==3.2.1", 
        "scipy==1.7.0", 
        "albumentations==0.5.2", 
        "dataclasses", 
        "pandas==1.2.1", 
        "torch==1.8.1",  
        "torchvision==0.9.1",
    ],
    long_description=readme,
    long_description_content_type = "text/markdown",
    author="Division of Medical Image Computing, German Cancer Research Center",
    maintainer_email="s.schuhegger@dkfz-heidelberg.de",
    entry_points={
        "console_scripts": [
            "bpreg_predict = bpreg.scripts.bpreg_inference:main",
        ]
    },
)
