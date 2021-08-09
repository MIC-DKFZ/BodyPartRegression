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

import sys, os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

sys.path.append("../../")
from bpreg.dataset.base_dataset import get_slices
from bpreg.evaluation.landmark_mse import LMSE
from bpreg.evaluation.accuracy import Accuracy
from bpreg.utils.linear_transformations import *
from bpreg.settings.settings import *

import json


class LandmarkScores:
    def __init__(
        self,
        data_path: str,
        df: pd.DataFrame,
        model: pl.LightningModule,
        device: str = "cuda",
        drop_cols=["val", "train", "test"],
        landmark_start: str = "pelvis_start",
        landmark_end: str = "eyes_end",
    ):

        self.data_path = data_path
        self.device = device
        self.model = model

        # expect filenames to be with or without .npy ending
        self.filenames = [
            f.replace(".npy", "") + ".npy" for f in df["filename"] if isinstance(f, str)
        ]
        self.filepaths = [os.path.join(data_path, f) for f in self.filenames]
        self.landmark_names = [
            l for l in df.columns if not l in drop_cols + ["filename"]
        ]

        self.index_matrix = np.array(
            df.drop(["filename"] + drop_cols, axis=1, errors="ignore")
        )
        self.score_matrix = self.create_score_matrix()

        self.expected_scores = np.nanmean(self.score_matrix, axis=0)
        self.expected_scores_std = np.nanstd(self.score_matrix, axis=0, ddof=1)

        self.expected_scores_transformed = self.transform(self.expected_scores.copy())
        self.score_matrix_transformed = self.transform(self.score_matrix.copy())

        self.lookuptable = self.create_lookuptable()
        if isinstance(landmark_start, float) and np.isnan(landmark_start):
            landmark_start = get_min_keyof_lookuptable(self.lookuptable)
        if isinstance(landmark_end, float) and np.isnan(landmark_end):
            landmark_end = get_max_keyof_lookuptable(self.lookuptable)
        self.transformed_lookuptable = transform_lookuptable(
            self.lookuptable, landmark_start=landmark_start, landmark_end=landmark_end
        )

    def create_score_matrix(self):
        with torch.no_grad():
            self.model.eval()
            self.model.to(self.device)
            slice_score_matrix = np.full(self.index_matrix.shape, np.nan)
            for i in np.arange(len(self.index_matrix)):
                filepath = self.filepaths[i]
                idxs = self.index_matrix[i, :]
                not_isnan = np.where(~np.isnan(idxs))
                idxs = idxs[not_isnan].astype(int)
                X = get_slices(filepath, idxs)
                y = self.model.predict_npy_array(X)
                slice_score_matrix[i, not_isnan] = y

        return slice_score_matrix

    def transform(self, x):
        min_value = self.expected_scores[0]
        max_value = self.expected_scores[-1]
        return linear_transform(x, scale=100, min_value=min_value, max_value=max_value)

    def create_lookuptable(self):
        lookuptable = {l: {} for l in self.landmark_names}

        for i, l in enumerate(self.landmark_names):
            lookuptable[l]["mean"] = self.expected_scores[i]
            lookuptable[l]["std"] = self.expected_scores_std[i]
        return lookuptable

    def print_lookuptable(self):
        for landmark, values in self.transformed_lookuptable.items():
            mean = np.round(values["mean"], 3)
            std = np.round(values["std"], 3)
            print(f"{landmark:<15}:\t {mean} +- {std}")

    def save_lookuptable(self, filepath):
        jsonDict = {
            "original": self.lookuptable,
            "transformed": self.transformed_lookuptable,
        }
        with open(filepath, "w") as f:
            json.dump(jsonDict, f, indent=4)


class LandmarkScoreBundle:
    def __init__(
        self,
        data_path,
        landmark_path,
        model,
        landmark_start="pelvis_start",
        landmark_end="eyes_end",
    ):
        df_database = pd.read_excel(landmark_path, sheet_name="database")
        df_train = pd.read_excel(landmark_path, sheet_name="landmarks-train")
        df_val = pd.read_excel(landmark_path, sheet_name="landmarks-val")
        df_test = pd.read_excel(landmark_path, sheet_name="landmarks-test")

        self.dict = {
            "validation": LandmarkScores(
                data_path,
                df_val,
                model,
                landmark_start=landmark_start,
                landmark_end=landmark_end,
            ),
            "train": LandmarkScores(
                data_path,
                df_train,
                model,
                landmark_start=landmark_start,
                landmark_end=landmark_end,
            ),
            "test": LandmarkScores(
                data_path,
                df_test,
                model,
                landmark_start=landmark_start,
                landmark_end=landmark_end,
            ),
            "train+val-all-landmarks": LandmarkScores(
                data_path,
                df_database[(df_database.train == 1) | (df_database.val == 1)],
                model,
                landmark_start=landmark_start,
                landmark_end=landmark_end,
            ),
            "test-all-landmarks": LandmarkScores(
                data_path,
                df_database[(df_database.test == 1)],
                model,
                landmark_start=landmark_start,
                landmark_end=landmark_end,
            ),
        }
        self.lmse = LMSE()
        self.model = model

    def nMSE(self, target="validation", reference="train"):
        lmse, lmse_std = self.lmse.from_matrices(
            self.dict[target].score_matrix, self.dict[reference].score_matrix
        )
        return lmse, lmse_std

    def nMSE_per_volume(self, target="validation", reference="train"):
        lmse, lmse_std = self.lmse.lmse_per_volume_from_matrices(
            self.dict[target].score_matrix, self.dict[reference].score_matrix
        )
        return lmse, lmse_std

    def accuracy(
        self, target_dataset, reference="train", class2landmark=CLASS_TO_LANDMARK_5
    ):
        acc = Accuracy(self.dict[reference].expected_scores, class2landmark)
        myAccuracy = acc.from_dataset(self.model, target_dataset)
        return myAccuracy

    def nMSE_per_landmark(self, target="validation", reference="train"):
        score_matrix = self.dict[target].score_matrix
        reference_matrix = self.dict[reference].score_matrix
        expected_scores = self.dict[reference].expected_scores
        d = self.lmse.get_normalizing_constant(expected_scores)
        landmark_names = self.dict[reference].landmark_names

        lmse_per_lanmdark = {landmark_name: {} for landmark_name in landmark_names}
        lmses, lmses_errors = self.lmse.lmse_per_landmark_from_matrices(
            score_matrix, reference_matrix
        )

        for landmark, lmse, lmse_std in zip(landmark_names, lmses, lmses_errors):
            lmse_per_lanmdark[landmark]["mean"] = lmse
            lmse_per_lanmdark[landmark]["std"] = lmse_std

        return lmse_per_lanmdark


def get_max_keyof_lookuptable(myDict):
    max_key = ""
    max_key_value = -np.inf
    for key in myDict:
        if myDict[key]["mean"] > max_key_value:
            max_key = key
            max_key_value = myDict[key]["mean"]
    return max_key


def get_min_keyof_lookuptable(myDict):
    min_key = ""
    min_key_value = +np.inf
    for key in myDict:
        if myDict[key]["mean"] < min_key_value:
            min_key = key
            min_key_value = myDict[key]["mean"]
    return min_key
