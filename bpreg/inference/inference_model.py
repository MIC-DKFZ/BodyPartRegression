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

from logging import warn
import numpy as np
import os, sys
import torch
import json, pickle
import argparse

sys.path.append("../../")

from bpreg.preprocessing.nifti2npy import Nifti2Npy
from bpreg.network_architecture.bpr_model import BodyPartRegression
from bpreg.score_processing import Scores, BodyPartExaminedDict
from bpreg.settings.settings import *
from bpreg.settings.model_settings import ModelSettings
from bpreg.score_processing.bodypartexamined_tag import *
from bpreg.utils.json_parser import *
from bpreg.scripts.initialize_pretrained_model import initialize_pretrained_model


from dataclasses import dataclass
from tqdm import tqdm


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

    def __init__(
        self,
        base_dir: str = DEFAULT_MODEL,
        gpu: bool = 1,
        warning_to_error: bool = False,
    ):

        self.base_dir = base_dir
        self.device = "cpu"
        if gpu:
            self.device = "cuda"

        self.model = load_model(base_dir, device=self.device)
        self.load_inference_settings()

        self.n2n = Nifti2Npy(
            target_pixel_spacing=3.5, min_hu=-1000, max_hu=1500, size=128
        )
        self.warning_to_error = warning_to_error

    def load_inference_settings(self):

        path = self.base_dir + "inference-settings.json"
        if not os.path.exists(path):
            print("WARNING: For this model, no inference settings can be load!")

        with open(path, "rb") as f:
            settings = json.load(f)

        # use for inference the lookuptable from all predictions
        # of the annotated landmarks in the train- and validation-dataset
        self.lookuptable_original = settings["lookuptable_train_val"]["original"]
        self.lookuptable = settings["lookuptable_train_val"]["transformed"]

        self.start_landmark = settings["settings"]["start-landmark"]
        self.end_landmark = settings["settings"]["end-landmark"]

        self.transform_min = self.lookuptable_original[self.start_landmark]["mean"]
        self.transform_max = self.lookuptable_original[self.end_landmark]["mean"]

        self.slope_mean = settings["slope_mean"]
        self.tangential_slope_min = settings["lower_quantile_tangential_slope"]
        self.tangential_slope_max = settings["upper_quantile_tangential_slope"]

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

    def predict_nifti(self, nifti_path: str):
        # get nifti file as tensor
        try:
            x, pixel_spacings = self.n2n.preprocess_nifti(nifti_path)
        except:
            x, pixel_spacings = np.nan, np.nan

        if isinstance(x, float) and np.isnan(x):
            x, pixel_spacings = self.n2n.load_volume(nifti_path)
            if not isinstance(x, np.ndarray):
                if self.warning_to_error:
                    raise ValueError(f"File {nifti_path} can not be loaded.")
                return np.nan

            warning_msg = (
                f"File {nifti_path.split('/')[-1]} with shape {x.shape} and pixel spacings {pixel_spacings} can not be converted to a 3-dimensional volume "
                + f"of the size {self.n2n.size}x{self.n2n.size}xz;"
            )
            print("WARNING: ", warning_msg)
            if self.warning_to_error:
                raise ValueError(warning_msg)
            return np.nan

        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x)
        x_tensor.to(self.device)

        # predict slice-scores
        scores = self.predict_tensor(x_tensor)
        return self.parse_scores(scores, pixel_spacings[2])

    def parse_scores(self, scores_array, pixel_spacing):

        scores = Scores(
            scores_array,
            pixel_spacing,
            transform_min=self.lookuptable_original[self.start_landmark]["mean"],
            transform_max=self.lookuptable_original[self.end_landmark]["mean"],
            slope_mean=self.slope_mean,
            tangential_slope_min=self.tangential_slope_min,
            tangential_slope_max=self.tangential_slope_max,
        )
        return scores

    def npy2json(
        self,
        X_: np.array,
        output_path: str,
        pixel_spacings: tuple,
        axis_ordering=(0, 1, 2),
        ignore_invalid_z: bool = False,
    ):
        """
        Method to predict slice scores from numpy arrays (in Hounsfiel dunits).
        Converts plain numpy array to numpy arrays which can be used by the DEFAULT_MODEL to predict the slice scores.n

        Args:
            X (np.array): matrix of CT volume in Hounsfield units.
            output_path (str): output path to save json file
            pixel_spacing (tuple): pixel spacing in x, y and z direction.
            axis_ordering (tuple): Axis ordering of CT volume. (0,1,2) is equivalent to the axis ordering xyz.
            ignore_invalid_z (bool): If true, than invalid z-spacing will be ignored for predicting the body part examined and not NONE will be given back.
        """
        X = self.n2n.preprocess_npy(X_, pixel_spacings, axis_ordering=axis_ordering)

        # convert axis ordering to zxy
        X = X.transpose(2, 0, 1)

        slice_scores = self.predict_npy_array(X)
        slice_scores = self.parse_scores(slice_scores, pixel_spacings[2])
        data_storage = VolumeStorage(
            slice_scores, self.lookuptable, ignore_invalid_z=ignore_invalid_z
        )
        if len(output_path) > 0:
            data_storage.save_json(output_path)
        return data_storage.json

    def nifti2json(
        self,
        nifti_path: str,
        output_path: str = "",
        stringify_json: bool = False,
        ignore_invalid_z: bool = False,
    ):
        """
        Main method to convert NIFTI CT volumes int JSON meta data files.
        Args:
            nifti_path (str): path of input NIFTI file
            output_path (str): output path to save JSON file
            stringify_json (bool): Set it to true for Kaapana JSON format
            axis_ordering (tuple): Axis ordering of CT volume. (0,1,2) is equivalent to the axis ordering xyz.
            ignore_invalid_z (bool): If true, than invalid z-spacing will be ignored for predicting the body part examined and not NONE will be given back.
        """
        slice_scores = self.predict_nifti(nifti_path)
        if isinstance(slice_scores, float) and np.isnan(slice_scores):
            return np.nan

        data_storage = VolumeStorage(
            slice_scores, self.lookuptable, ignore_invalid_z=ignore_invalid_z
        )
        if len(output_path) > 0:
            data_storage.save_json(output_path, stringify_json=stringify_json)
        return data_storage.json


@dataclass
class VolumeStorage:
    """Body part metadata for one volume

    Args:
        scores (Scores): predicted slice scores
        lookuptable (dict): reference table which contains expected scores for anatomies
        body_parts ([type], optional): dictionary to define the body parts for the tag: "body part examined". Defaults to BODY_PARTS.
        body_parts_included ([type], optional): dictionary to calculate the "body part examined tag". Defaults to BODY_PARTS_INCLUDED.
        distinct_body_parts ([type], optional): dictionary to calculate the "body part examined tag". Defaults to DISTINCT_BODY_PARTS.
        min_present_landmarks ([type], optional): dictionary to calculate the "body part examined rtag". Defaults to MIN_PRESENT_LANDMARKS.
    """

    def __init__(
        self,
        scores: Scores,
        lookuptable: dict,
        body_parts=BODY_PARTS,
        body_parts_included=BODY_PARTS_INCLUDED,
        distinct_body_parts=DISTINCT_BODY_PARTS,
        min_present_landmarks=MIN_PRESENT_LANDMARKS,
        ignore_invalid_z: bool = False,
    ):

        self.ignore_invalid_z = ignore_invalid_z
        self.body_parts = body_parts
        self.body_parts_included = body_parts_included
        self.distinct_body_parts = distinct_body_parts
        self.min_present_landmarks = min_present_landmarks

        self.cleaned_slice_scores = list(scores.values.astype(np.float64))
        self.z = list(scores.z.astype(np.float64))
        self.unprocessed_slice_scores = list(
            scores.original_transformed_values.astype(np.float64)
        )
        self.lookuptable = lookuptable

        self.zspacing = float(scores.zspacing)  # .astype(np.float64)
        self.reverse_zordering = float(scores.reverse_zordering)
        self.valid_zspacing = float(scores.valid_zspacing)
        self.expected_slope = float(scores.slope_mean)
        self.observed_slope = float(scores.a)
        self.expected_zspacing = float(scores.expected_zspacing)
        self.r_slope = float(scores.r_slope)
        self.bpe = BodyPartExaminedDict(lookuptable, body_parts=self.body_parts)
        self.bpet = BodyPartExaminedTag(
            lookuptable,
            body_parts_included=self.body_parts_included,
            distinct_body_parts=self.distinct_body_parts,
            min_present_landmarks=self.min_present_landmarks,
            ignore_invalid_z=self.ignore_invalid_z,
        )

        self.settings = {
            "slice score processing": scores.settings,
            "body part examined dict": self.body_parts,
            "body part examined tag": {
                "body parts included": self.body_parts_included,
                "distinct body parts": self.distinct_body_parts,
                "min present landmarks": self.min_present_landmarks,
            },
        }

        self.json = {
            "cleaned slice scores": self.cleaned_slice_scores,
            "z": self.z,
            "unprocessed slice scores": self.unprocessed_slice_scores,
            "body part examined": self.bpe.get_examined_body_part(
                self.cleaned_slice_scores
            ),
            "body part examined tag": self.bpet.estimate_tag(scores),
            "look-up table": self.lookuptable,
            "reverse z-ordering": self.reverse_zordering,
            "valid z-spacing": self.valid_zspacing,
            "expected slope": self.expected_slope,
            "observed slope": self.observed_slope,
            "slope ratio": self.r_slope,
            "expected z-spacing": self.expected_zspacing,
            "z-spacing": self.zspacing,
            "settings": self.settings,
        }

    def save_json(self, output_path: str, stringify_json=False):
        """Store data in json file

        Args:
            output_path (str): save path for json file
            stringify_json (bool, optional): if True, stringify output of parameters and
            convert json file to a Kaapana friendly format
        """
        data = self.json
        if stringify_json:
            data = parse_json4kaapana(data)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)


def load_model(
    base_dir, model_file="model.pt", config_file="config.json", device="cuda"
):
    # load public model, if it does not exist locally
    if (base_dir == DEFAULT_MODEL) & ~os.path.exists(base_dir):
        initialize_pretrained_model()

    config_filepath = base_dir + config_file
    model_filepath = base_dir + model_file

    config = ModelSettings()
    config.load(path=config_filepath)

    model = BodyPartRegression(alpha=config.alpha, lr=config.lr)
    model.load_state_dict(
        torch.load(model_filepath, map_location=torch.device(device)), strict=False
    )
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

    base_dir = "../../src/models/private_bpr_model/"
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
