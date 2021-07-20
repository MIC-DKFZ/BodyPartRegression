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
import torch


sys.path.append("../")

from bpreg.inference.inference_model import InferenceModel
from bpreg.settings import *


def bpreg_for_directory(model_path: str, input_dirpath: str, output_dirpath: str, skip_existing: bool=1):
    # test if gpu is available 
    gpu_available = torch.cuda.is_available()

    model = InferenceModel(model_path, gpu=gpu_available)
    ifiles = [f for f in os.listdir(input_dirpath) if f.endswith((".nii.gz", ".nii"))]
    ofiles = [f.replace(".nii", "").replace(".gz", "") + ".json" for f in ifiles]

    for ifile, ofile in zip(ifiles, ofiles):
        ipath = input_dirpath + ifile
        opath = output_dirpath + ofile
        if os.path.exists(opath) and skip_existing==1:
            print(f"JSON-file already exists. Skip file: {opath}")
            continue
        print(f"Create body-part meta data file: {opath}")
        model.nifti2json(ipath, opath)


def main():
    # default_model = "../src/models/private_bpr_model/" # TODO 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-i", default="")
    parser.add_argument("-o", default="")
    parser.add_argument("--skip", default=1)
    # TODO --plot 0/1 
    # TODO report 

    value = parser.parse_args()
    model_path = value.model
    input_dirpath = value.i
    output_dirpath = value.o
    skip_existing = value.skip

    bpreg_for_directory(model_path, input_dirpath, output_dirpath, skip_existing=skip_existing)


if __name__ == "__main__":
    main()
