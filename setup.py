import os
from setuptools import setup, find_packages
from setuptools.command.install import install


def resolve_requirements(file):
    requirements = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                requirements += resolve_requirements(
                    os.path.join(os.path.dirname(file), r.split(" ")[1])
                )
            else:
                requirements.append(r)
    return requirements


# load requirements
requirements = resolve_requirements(
    os.path.join(os.path.dirname(__file__), "requirements.txt")
)

# read readme
with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="bpreg",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    test_suite="unittest",
    install_requires=requirements,
    long_description=readme,
    author="Division of Medical Image Computing, German Cancer Research Center",
    maintainer_email="",
    entry_points={
        "console_scripts": [
            "bpreg_predict = scripts.bpreg_inference:main",
        ]
    },
)
