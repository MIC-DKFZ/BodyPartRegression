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

import os, sys
import argparse


from tqdm import tqdm


sys.path.append("../")

from scripts.inference.inference_model import InferenceModel
from scripts.settings.settings import *


def bpreg_for_directory(model_path: str, input_dirpath: str, output_dirpath: str):
    model = InferenceModel(model_path)
    ifiles = [f for f in os.listdir(input_dirpath) if f.endswith((".nii.gz", ".nii"))]
    ofiles = [f.replace(".nii", "").replace(".gz", "") + ".json" for f in ifiles]

    for ifile, ofile in tqdm(zip(ifiles, ofiles)):
        ipath = input_dirpath + ifile
        opath = output_dirpath + ofile
        if os.path.exists(opath):
            continue
        model.nifti2json(input_dirpath + ifile, output_dirpath + ofile)


if __name__ == "__main__":
    default_model = "../src/models/loh/version_1/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--i", default="")
    parser.add_argument("--o", default="")

    value = parser.parse_args()
    model_path = value.model
    input_dirpath = value.i
    output_dirpath = value.o

    bpreg_for_directory(model_path, input_dirpath, output_dirpath)
