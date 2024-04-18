from setuptools import setup, find_packages

# read readme
with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="bpreg",
    version="1.1.0",
    packages=find_packages(),
    url="https://github.com/MIC-DKFZ/BodyPartRegression",
    include_package_data=True,
    package_data={"bpreg": ["settings/body-part-metadata.md"]},
    test_suite="unittest",
    install_requires=[
        "pytorch_lightning>=1.2.10",
        "nibabel>=3.2.1",
        "scipy>=1.7.0",
        # 1.4.4 includes https://github.com/albumentations-team/albumentations/pull/1646.
        # This disallows passing value=None to ShiftScaleRotate.
        # The public model version 1.1 sets value=null in its config.json: https://zenodo.org/records/5113483#.YPaBkNaxWEA
        "albumentations>=0.5.2,<1.4.4",
        "dataclasses; python_version < '3.7'",
        "matplotlib",
        "pandas>=1.2.1",
        "requests",
        "torch>=1.8.1",
        "torchvision>=0.9.1 ",
    ],
    data_files=[("models", ["bpreg/settings/body-part-metadata.md"])],
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Division of Medical Image Computing, German Cancer Research Center",
    author_email="mic-office@dkfz.de",
    maintainer_email="s.schuhegger@dkfz-heidelberg.de",
    entry_points={
        "console_scripts": [
            "bpreg_predict = bpreg.scripts.bpreg_inference:main",
        ]
    },
)
