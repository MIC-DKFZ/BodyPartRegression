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
from scripts.initialize_pretrained_model import initialize_pretrained_model
import argparse
import torch


sys.path.append("../")

from bpreg.inference.inference_model import InferenceModel
from bpreg.settings import *


def bpreg_for_directory(
    model_path: str,
    input_dirpath: str,
    output_dirpath: str,
    skip_existing: bool = True,
    stringify_json: bool = False,
):
    # test if gpu is available
    gpu_available = torch.cuda.is_available()

    model = InferenceModel(model_path, gpu=gpu_available)
    ifiles = [f for f in os.listdir(input_dirpath) if f.endswith((".nii.gz", ".nii"))]
    ofiles = [f.replace(".nii", "").replace(".gz", "") + ".json" for f in ifiles]

    for ifile, ofile in zip(ifiles, ofiles):
        ipath = os.path.join(input_dirpath, ifile)
        opath = os.path.join(output_dirpath, ofile)
        if os.path.exists(opath) and skip_existing == 1:
            print(f"JSON-file already exists. Skip file: {ifile}")
            continue
        print(f"Create body-part meta data file: {ofile}")
        model.nifti2json(ipath, opath, stringify_json=stringify_json)


def bpreg_inference(
    input_path: str,
    output_path: str,
    model: str,
    skip_existing: bool,
    stringify_json: bool,
):
    # load public model, if it does not exists locally
    if (model == DEFAULT_MODEL) & ~os.path.exists(model):
        initialize_pretrained_model()

    # run body part regression for each file in the dictionary
    bpreg_for_directory(
        model,
        input_path,
        output_path,
        skip_existing=skip_existing,
        stringify_json=stringify_json,
    )


def main():
    # default_model = "../src/models/private_bpr_model/" # TODO

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-i", default="")
    parser.add_argument("-o", default="")
    parser.add_argument("--skip", default=True)
    parser.add_argument("--str", default=False)

    # TODO --plot 0/1
    # TODO report

    value = parser.parse_args()
    model_path = value.model
    input_dirpath = value.i
    output_dirpath = value.o
    skip_existing = value.skip
    stringify_json = value.str

    bpreg_inference(
        input_dirpath, output_dirpath, model_path, skip_existing, stringify_json
    )


if __name__ == "__main__":
    main()
