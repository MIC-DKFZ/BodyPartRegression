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
        "pytorch_lightning==2.2.0",
        "nibabel==5.3.0",
        "scipy==1.16.0",
        "albumentations==1.4.1",
        "dataclasses",
        "numpy==2.2.6",
        "pandas==2.2.2",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "requests==2.32.3",
        "pynrrd==1.1.1",
        "matplotlib"
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
