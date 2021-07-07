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

import numpy as np
import os, sys
import torch
import json, pickle
import argparse

sys.path.append("../../")

from bpreg.preprocessing.nifti2npy import Nifti2Npy
from bpreg.network_architecture.bpr_model import BodyPartRegression
from bpreg.score_processing.scores import Scores
from bpreg.score_processing.landmark_scores import (
    get_max_keyof_lookuptable,
    get_min_keyof_lookuptable,
)
from bpreg.score_processing.bodypartexamined import BodyPartExamined

from dataclasses import dataclass
from tqdm import tqdm


# TODO predict_tensor sonst überall rausnehmen
# TODO predict_npy_array rausnehmen aus base_model
# TODO create Tests to test load_model and InferenceModel


class InferenceModel:
    """
    Body Part Regression Model for inference purposes.

    Args:
        base_dir (str]): Path which includes model related file.
        Structure of base_dir:
        base_dir/
            model.pt - includes model
            settings.json - includes mean slope and mean slope std
            lookuptable.json - includes lookuptable as reference
        device (str, optional): [description]. "cuda" or "cpu"
    """

    def __init__(self, base_dir: str, gpu: int = 1):

        self.base_dir = base_dir
        self.device = "cpu"
        if gpu:
            self.device = "cuda"

        self.model = load_model(base_dir, device=self.device)
        self.load_lookuptable()
        self.load_settings()
        self.n2n = Nifti2Npy(
            target_pixel_spacing=3.5, min_hu=-1000, max_hu=1500, size=128
        )

    def load_lookuptable(self):
        path = self.base_dir + "lookuptable.json"
        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            lookuptable = json.load(f)

        try:
            self.lookuptable_original = lookuptable["original"]
            self.lookuptable = lookuptable["transformed"]
            if "start-landmark" in lookuptable.keys():
                self.start_landmark = lookuptable["start-landmark"]
            else:
                self.start_landmark = get_min_keyof_lookuptable(self.lookuptable)

            if "end-landmark" in lookuptable.keys():
                self.end_landmark = lookuptable["end-landmark"]
            else:
                self.end_landmark = get_max_keyof_lookuptable(self.lookuptable)

            self.transform_min = lookuptable["original"][self.start_landmark]["mean"]
            self.transform_max = lookuptable["original"][self.end_landmark]["mean"]

        except:

            return

    def load_settings(self):
        path = self.base_dir + "settings.json"
        if not os.path.exists(path):
            return

        with open(path, "rb") as f:
            mySettings = json.load(f)

        self.slope_mean = mySettings["slope_mean"]
        self.slope_std = mySettings["slope_std"]

    def predict_tensor(self, tensor, n_splits=200):
        scores = []
        n = tensor.shape[0]
        slice_splits = list(np.arange(0, n, n_splits))
        slice_splits.append(n)

        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            for i in range(len(slice_splits) - 1):
                min_index = slice_splits[i]
                max_index = slice_splits[i + 1]
                score = self.model(tensor[min_index:max_index, :, :, :].to(self.device))
                scores += [s.item() for s in score]

        scores = np.array(scores)
        return scores

    def predict_npy_array(self, x, n_splits=200):
        x_tensor = torch.tensor(x[:, np.newaxis, :, :]).to(self.device)
        scores = self.predict_tensor(x_tensor, n_splits=n_splits)
        return scores

    def predict_nifti(self, nifti_path):
        # get nifti file as tensor
        x, pixel_spacings = self.n2n.preprocess_nifti(nifti_path)
        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x)
        x_tensor.to(self.device)

        # predict slice-scores
        scores = self.predict_tensor(x_tensor)
        return self.parse_scores(scores, pixel_spacings[2])

    def estimated_slope(self, dataset):
        slopes = []
        for i in tqdm(range(len(dataset))):
            X = dataset.get_full_volume(i)
            z = dataset.z_spacings[i]
            scores = self.predict_npy_array(X)
            myScores = self.parse_scores(scores, z)
            slopes.append(myScores.a)

        slopes = np.array(slopes)
        return np.nanmean(slopes), np.nanstd(slopes, ddof=1)

    def parse_scores(self, scores_array, pixel_spacing):

        scores = Scores(
            scores_array,
            pixel_spacing,
            transform_min=self.lookuptable_original[self.start_landmark]["mean"],
            transform_max=self.lookuptable_original[self.end_landmark]["mean"],
            slope_mean=self.slope_mean,
            slope_std=self.slope_std,
        )
        return scores

    def npy2json(self, X, output_path, pixel_spacing):
        slice_scores = self.predict_npy_array(X)
        slice_scores = self.parse_scores(slice_scores, pixel_spacing)
        data_storage = VolumeStorage(slice_scores, self.lookuptable)
        if len(output_path) > 0:
            data_storage.save_json(output_path)
        return data_storage.json

    def nifti2json(self, nifti_path, output_path):
        slice_scores = self.predict_nifti(nifti_path)
        data_storage = VolumeStorage(slice_scores, self.lookuptable)
        if len(output_path) > 0:
            data_storage.save_json(output_path)
        return data_storage.json


@dataclass
class VolumeStorage:
    def __init__(self, scores: Scores, lookuptable: dict):
        self.cleaned_slice_scores = list(scores.values.astype(np.float64))
        self.z = list(scores.z.astype(np.float64))
        self.unprocessed_slice_scores = list(
            scores.original_transformed_values.astype(np.float64)
        )
        self.lookuptable = lookuptable

        self.zspacing = float(scores.zspacing)  # .astype(np.float64)
        self.reverse_zordering = float(scores.reverse_zordering)
        self.valid_zspacing = float(scores.valid_zspacing)

        self.bpe = BodyPartExamined(lookuptable)

        self.json = {
            "cleaned slice scores": self.cleaned_slice_scores,
            "z": self.z,
            "unprocessed slice scores": self.unprocessed_slice_scores,
            "body part examined": self.bpe.get_examined_body_part(
                self.cleaned_slice_scores
            ),
            "look-up table": self.lookuptable,
            "reverse z-ordering": self.reverse_zordering,
            "valid z-spacing": self.valid_zspacing,
            "z-spacing": self.zspacing,
        }

    def save_json(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.json, f)


def load_model(base_dir, model_file="model.pt", config_file="config.p", device="cuda"):
    config_filepath = base_dir + config_file  # TODO
    model_filepath = base_dir + model_file  # TODO verallgemeinern

    with open(config_filepath, "rb") as f:
        config = pickle.load(f)

    model = BodyPartRegression(alpha=config["alpha"], lr=config["lr"])
    model.load_state_dict(torch.load(model_filepath, map_location=torch.device(device)))
    model.eval()
    model.to(device)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", default="")
    parser.add_argument("--o", default="")
    parser.add_argument("--g", default=1)

    value = parser.parse_args()
    ipath = value.i
    opath = value.o
    gpu = value.g

    base_dir = "../../src/models/loh/version_1/"
    model = InferenceModel(base_dir, gpu=gpu)

    data_path = "../../data/test_cases/"
    nifti_paths = [
        data_path + f for f in os.listdir(data_path) if f.endswith(".nii.gz")
    ]
    for nifti_path in tqdm(nifti_paths):
        output_path = nifti_path.replace("test_cases", "test_results").replace(
            ".nii.gz", ".json"
        )
        model.nifti2json(nifti_path, output_path)
