import os, sys
import argparse


from tqdm import tqdm


sys.path.append("../")

from bpreg.inference.inference_model import InferenceModel
from bpreg.settings.settings import *


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
