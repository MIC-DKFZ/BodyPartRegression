import os
from setuptools import setup, find_packages
from setuptools.command.install import install

PUBLIC_MODEL_URL = (
    "https://zenodo.org/record/5113483/files/public_bpr_model.zip?download=1"
)
SAVE_PUBLIC_MODEL_PATH = "src/models/public_inference_model.zip"


class PostInstallCommand(install): 

    def run(self): 
        install.run()
        import requests, zipfile 

        # Download public model from zenodo, for inference
        print("Download publicly available body part regression model.")

        with requests.get(PUBLIC_MODEL_URL, stream="True") as r:
            r.raise_for_status()
            with open(SAVE_PUBLIC_MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=None):
                    f.write(chunk)

        # unzip public model
        save_path = "/".join(SAVE_PUBLIC_MODEL_PATH.split("/")[:-1]) + "/"
        with zipfile.ZipFile(SAVE_PUBLIC_MODEL_PATH, "r") as f:
            f.extractall(save_path)

        # remove file
        os.remove(SAVE_PUBLIC_MODEL_PATH)

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
    entry_points={"console_scripts": ["bpreg_predict = scripts.bpreg_inference:main"]},
    cmdclass={'install': PostInstallCommand}
)


