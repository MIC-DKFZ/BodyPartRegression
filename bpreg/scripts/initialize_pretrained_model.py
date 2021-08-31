"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import requests, zipfile, os
from bpreg.settings.settings import MAIN_PATH


PUBLIC_MODEL_URL = (
    "https://zenodo.org/record/5113483/files/public_bpr_model.zip?download=1"
)

SAVE_PUBLIC_MODEL_DIR_PATH = os.path.join(MAIN_PATH, "src/models/")
SAVE_PUBLIC_MODEL_PATH = os.path.join(
    SAVE_PUBLIC_MODEL_DIR_PATH, "public_inference_model.zip"
)


def initialize_pretrained_model():
    # Download public model from zenodo, for inference
    print(
        "Download publicly available body part regression model from Zenodo:\n",
        "https://zenodo.org/record/5113483",
    )

    with requests.get(PUBLIC_MODEL_URL, stream="True") as r:
        r.raise_for_status()
        if not os.path.exists(os.path.join(MAIN_PATH, "src")):
            os.mkdir(os.path.join(MAIN_PATH, "src"))
        if not os.path.exists(os.path.join(MAIN_PATH, "src/models/")):
            os.mkdir(os.path.join(MAIN_PATH, "src/models/"))

        with open(SAVE_PUBLIC_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=None):
                f.write(chunk)

    # unzip public model
    save_path = os.path.dirname(SAVE_PUBLIC_MODEL_PATH)
    with zipfile.ZipFile(SAVE_PUBLIC_MODEL_PATH, "r") as f:
        f.extractall(save_path)

    # remove file
    os.remove(SAVE_PUBLIC_MODEL_PATH)


if __name__ == "__main__":
    initialize_pretrained_model()
