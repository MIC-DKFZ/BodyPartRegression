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
from shutil import copyfile

sys.path.append("../")
from bpreg.inference.inference_model import InferenceModel
from bpreg.settings import *
from bpreg.evaluation.visualization import plot_scores


def bpreg_for_directory(
    model_path: str,
    input_dirpath: str,
    output_dirpath: str,
    skip_existing: bool = True,
    stringify_json: bool = False,
    gpu_available: bool = True,
):
    # test if gpu is available
    if not torch.cuda.is_available():
        gpu_available = False
    model = InferenceModel(model_path, gpu=gpu_available)

    if input_dirpath == "":
        raise ValueError(
            "Input path is not defined. Please define the input path via -i <input_path>."
        )

    if output_dirpath == "":
        raise ValueError(
            "Output path is not defined. Please define the output path via -o <output_path>."
        )

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


def plot_scores_in_json_files(output_path):
    json_files = [f for f in os.listdir(output_path) if f.endswith(".json")]
    for file in json_files:
        save_path = os.path.join(output_path, file.replace(".json", ".png"))
        plot_scores(os.path.join(output_path, file), save_path=save_path)


def bpreg_inference(
    input_path: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    skip_existing: bool = True,
    stringify_json: bool = False,
    gpu_available: bool = True,
    plot: bool = False,
):

    # run body part regression for each file in the dictionary
    bpreg_for_directory(
        model,
        input_path,
        output_path,
        skip_existing=skip_existing,
        stringify_json=stringify_json,
        gpu_available=gpu_available,
    )

    # plot slice scores if plot is True
    if plot:
        print("Plot slice scores. ")
        plot_scores_in_json_files(output_path)

    # copy documentation for json metadata files into repository
    copyfile(
        os.path.join(MAIN_PATH, "bpreg/settings/body-part-metadata.md"),
        os.path.join(output_path, "README.md"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--plot", default=False)
    parser.add_argument("-i", default="")
    parser.add_argument("-o", default="")
    parser.add_argument("--skip", default=True)
    parser.add_argument("--str", default=False)
    parser.add_argument("--gpu", default=True)

    value = parser.parse_args()
    model_path = value.model
    plot = value.plot
    input_dirpath = value.i
    output_dirpath = value.o
    skip_existing = value.skip
    stringify_json = value.str
    gpu_available = value.gpu

    bpreg_inference(
        input_dirpath,
        output_dirpath,
        model_path,
        skip_existing,
        stringify_json,
        gpu_available=gpu_available,
        plot=plot,
    )


if __name__ == "__main__":
    main()
