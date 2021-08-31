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

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append("../../")
from bpreg.utils.training_utils import get_dataframe, get_datasets
from bpreg.evaluation.landmark_mse import LMSE
from bpreg.evaluation.visualization import Visualization
from bpreg.inference.inference_model import InferenceModel
from bpreg.score_processing import LandmarkScoreBundle
from bpreg.settings import *


class Evaluation(Visualization):
    """evaluate body part regression model

    Args:
        base_filepath (str): path of trained model
        df_data_source_path (str, optional): Path to dataframe which defines train/val/test split. Defaults to DF_DATA_SOURCE_PATH.
        landmark_path (str, optional): Path to dataframe which saves the annotated landmarks. Defaults to LANDMARK_PATH.
        data_path (str, optional): Path where the .npy files for training are saved. Defaults to DATA_PATH.
        device (str, optional): device, where to save the model. Defaults to "cuda".
        landmark_start (str, optional): start landmark gets mapped through transformation to zero. Defaults to "pelvis_start".
        landmark_end (str, optional): end_landmark gets mapped through transformation to 100. Defaults to "eyes_end".
    """

    def __init__(
        self,
        base_filepath: str,
        df_data_source_path: str = DF_DATA_SOURCE_PATH,
        landmark_path: str = LANDMARK_PATH,
        data_path: str = DATA_PATH,
        device: str = "cuda",
        landmark_start: str = "pelvis_start",
        landmark_end: str = "eyes_end",
    ):

        Visualization.__init__(self)
        self.device = device
        self.base_filepath = base_filepath
        self.inference_model = InferenceModel(base_filepath)
        self.normalizedMSE = LMSE()
        self.landmark_score_bundle = LandmarkScoreBundle(
            data_path,
            landmark_path,
            self.inference_model.model,
            landmark_start=landmark_start,
            landmark_end=landmark_end,
        )

        # setup data
        self.df_data_source_path = df_data_source_path
        self.landmark_path = landmark_path
        self.data_path = data_path
        self._setup_data()

        self.mse, self.mse_std = self.landmark_score_bundle.nMSE(
            target="validation", reference="train"
        )
        self.acc5 = self.landmark_score_bundle.accuracy(
            self.val_dataset, reference="train", class2landmark=CLASS_TO_LANDMARK_5
        )

    def _setup_data(self):
        path = self.base_filepath + "config.json"

        self.config = ModelSettings()
        self.config.load(path)

        config = self.config
        config.num_slices = 8
        config.batch_size = 32
        config.shuffle_train_dataloader = False

        if len(self.df_data_source_path) > 0:
            config.df_data_source_path = self.df_data_source_path
        if len(self.landmark_path) > 0:
            config.landmark_path = self.landmark_path
        if len(self.data_path) > 0:
            config.data_path = self.data_path

        df_data = get_dataframe(config)

        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(
            config, df_data
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            num_workers=20,
            shuffle=config.shuffle_train_dataloader,
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=config.batch_size, num_workers=20
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=config.batch_size, num_workers=20
        )

    def print_summary(self):
        print("Model summary\n*******************************")

        print(
            f"\nLandmark Mean Square Error:\t{self.mse:<1.3f} +- {self.mse_std:<1.3f}"
        )
        print(f"Accuracy (5 classes) :        \t{self.acc5*100:<1.2f}%")
        print("\nLook-up table for training data \n*******************************")
        self.landmark_score_bundle.dict["train"].print_lookuptable()

    def plot_landmarks(self, colors=[], alpha=0.7, target="validation"):
        if target == "validation":
            score_matrix = self.landmark_score_bundle.dict[
                "validation"
            ].score_matrix_transformed
        if target == "test":
            score_matrix = self.landmark_score_bundle.dict[
                "test"
            ].score_matrix_transformed
        expected_scores = self.landmark_score_bundle.dict[
            "train"
        ].expected_scores_transformed

        landmark_names = self.landmark_score_bundle.dict["validation"].landmark_names
        super(Evaluation, self).plot_landmarks(
            score_matrix,
            expected_scores=expected_scores,
            landmark_names=landmark_names,
            alpha=alpha,
            colors=colors,
        )

    def plot_slices2scores(
        self, max_cols=4, nearby_values=[0, 25, 50, 75, 100], save_path="", fontsize=16
    ):
        _, ax = plt.subplots(len(nearby_values), max_cols, figsize=(14, 16))

        for row, nearby_value in enumerate(nearby_values):
            col = 0
            slice_indices = []
            while col < max_cols:
                idx = np.random.randint(0, len(self.test_dataset))
                if idx in slice_indices:
                    continue
                slice_indices.append(idx)

                filepaths = self.test_dataset.filepaths
                zspacing = self.test_dataset.z_spacings[idx]
                X = np.load(filepaths[idx]).transpose(2, 0, 1)
                scores = self.inference_model.predict_npy_array(X)
                myScores = self.inference_model.parse_scores(scores, zspacing)
                slice_index = np.argmin(
                    np.abs(myScores.original_transformed_values - nearby_value)
                )
                slice_score = myScores.original_transformed_values[slice_index]
                if np.abs(slice_score - nearby_value) > 1:
                    continue

                ax[row, col].imshow(X[slice_index, :, :], cmap="gray")
                ax[row, col].set_title(np.round(slice_score, 2), fontsize=fontsize)
                ax[row, col].set_yticklabels([])
                ax[row, col].set_xticklabels([])
                ax[row, col].set_yticks([])
                ax[row, col].set_xticks([])
                col += 1
        plt.tight_layout()
        if len(save_path) > 0:
            plt.savefig(save_path)


if __name__ == "__main__":
    base_dir = "/home/AD/s429r/Documents/Code/bodypartregression/src/models/loh-ldist-l2/sigma-dataset-v11-v2/"
    modelEval = Evaluation(base_dir)
