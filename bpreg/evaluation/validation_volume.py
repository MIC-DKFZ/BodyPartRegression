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

from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

from bpreg.score_processing import Scores
from bpreg.utils.linear_transformations import *
from bpreg.dataset.base_dataset import BaseDataset


class ValidationVolume:
    def __init__(
        self,
        inference_model,
        val_dataset: BaseDataset,
        idx: int,
        expected_scores: np.array,
        fontsize: float = 22,
    ):
        self.landmark_idx = idx
        self.data_idx = val_dataset.landmark_ids[idx]

        self.landmarks = val_dataset.landmark_matrix[self.landmark_idx, :]
        self.landmark_names = val_dataset.landmark_names

        self.nonempty = np.where(~np.isnan(self.landmarks))[0]
        self.filename = val_dataset.filenames[self.data_idx]
        self.expected_scores = expected_scores
        self.X = val_dataset.get_full_volume(self.data_idx)
        self.z = val_dataset.z_spacings[self.data_idx]
        self.scores = inference_model.predict_npy_array(self.X)
        self.scores = Scores(
            self.scores,
            self.z,
            transform_min=expected_scores[0],
            transform_max=expected_scores[-1],
        )
        self.interpolated_x, self.interpolated_scores = self.get_interpolated_scores(
            self.landmarks, self.expected_scores
        )
        self.z_array = np.arange(0, len(self.X)) * self.z
        self.fontsize = fontsize
        self.figsize = (14, 8)
        self.markerstyles = np.array(
            [".", "v", "*", "p", "D", "X", "h", "P", "+", "^", "x", "d"]
        )
        self.colors = np.array(
            [
                "firebrick",
                "sienna",
                "tan",
                "olivedrab",
                "lightseagreen",
                "paleturquoise",
                "royalblue",
                "navy",
                "mediumpurple",
                "mediumvioletred",
                "crimson",
                "saddlebrown",
            ]
        )

    def plot_scores(self, set_figsize=False, legend=None):
        if set_figsize:
            plt.figure(figsize=self.figsize)
        label = None

        # plot scores
        plt.plot(
            self.z_array,
            self.scores.original_transformed_values,
            color="black",
            linewidth=1,
        )

        # plot landmarks
        for name, landmark, color, score in zip(
            self.landmark_names[self.nonempty],
            self.landmarks[self.nonempty],
            self.colors[self.nonempty],
            self.scores.original_transformed_values[
                self.landmarks[self.nonempty].astype(int)
            ],
        ):
            if legend:
                label = name
            plt.plot(
                landmark * self.z,
                score,
                linestyle="",
                marker=".",
                markersize=15,
                color=color,
                label=label,
            )

        plt.xlabel("z [mm]", fontsize=self.fontsize)
        plt.ylabel("Slice Score", fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize - 2)
        plt.yticks(fontsize=self.fontsize - 2)
        if legend:
            plt.legend(loc=0)

    def transform(self, x):
        return linear_transform(
            x,
            scale=100,
            min_value=self.expected_scores[0],
            max_value=self.expected_scores[-1],
        )

    def plot_expected_scores(
        self, ax=False, title=None, legend=True, ylim=(-10, 80), ylabel=True
    ):
        if not ax:
            fig, ax = plt.subplots(figsize=(self.figsize))
        ax.plot(
            self.z_array,
            self.scores.original_transformed_values,
            linewidth=2,
            label="predicted scores",
        )

        ax.plot(
            self.landmark_positions * self.z,
            self.landmark_scores,
            linestyle="",
            marker="+",
            markersize=15,
            color="#ff7f0e",
        )
        ax.plot(
            self.interpolated_x,
            self.interpolated_scores,
            linestyle="--",
            color="#ff7f0e",
            linewidth=2,
            label="expected scores",
        )
        if legend:
            ax.legend(fontsize=self.fontsize)
        ax.tick_params(labelsize=self.fontsize - 2)
        ax.tick_params(labelsize=self.fontsize - 2)
        ax.set_xlabel("z [mm]", fontsize=self.fontsize)
        if ylabel:
            ax.set_ylabel("Slice scores", fontsize=self.fontsize)
        ax.set_title(title, fontsize=self.fontsize)
        ax.set_ylim(ylim)

    def get_interpolated_scores(self, landmarks, expected_scores):
        nonempty = np.where(~np.isnan(landmarks))
        x = np.arange(np.nanmin(landmarks), np.nanmax(landmarks) + 1)

        self.landmark_positions = landmarks[nonempty]
        self.landmark_scores = self.transform(expected_scores[nonempty])
        self.defined_landmarks = self.landmark_names[nonempty]
        func = interpolate.interp1d(
            self.landmark_positions, self.landmark_scores, kind="linear"
        )
        interpolated_scores = func(x)

        return x * self.z, interpolated_scores
