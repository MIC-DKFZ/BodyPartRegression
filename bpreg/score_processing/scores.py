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

import torch
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.append("../../")
from bpreg.utils.linear_transformations import *


class Scores:
    """Scores and additional meta data inforamtion based on the prediction from the Body Part Regression (bpr) model.

    Args:
        scores (list): predictions from the bpr model.
        zspacing (float): zspacing of analzed volume.
        transform_min (float, optional): score which should get mapped to zero. Defaults to np.nan.
        transform_max (float, optional): score which should get mapped to 100. Defaults to np.nan.
        smoothing_sigma (float, optional): Smoothing sigma in mm, for gaussian smoothing of the scores. Defaults to 10.
        tangential_slope_min (float, optional): minimum valid tangential slope. Defaults to -0.037.
        tangential_slope_max (float, optional): maximum valid tangential slope. Defaults to 0.25.
        slope_mean (float, optional): expected slope of slice score curve. Defaults to np.nan.
        background_scores (list, optional): slice score prediction of empty slice. Defaults to [110.83, 6.14].
        r_slope_threshold (float, optional): threshold for declaring the z-spacing as invalid. Defaults to 0.28.
    """

    def __init__(
        self,
        scores: list,
        zspacing: float,
        transform_min: float,
        transform_max: float,
        smoothing_sigma: float = 10,
        tangential_slope_min: float = -0.037,
        tangential_slope_max: float = 0.25,
        slope_mean: float = np.nan,
        background_scores=[110.83, 6.14, 108.92, 6.22],
        r_slope_threshold: float = 0.28,
    ):

        scores = np.array(scores).astype(float)
        self.background_scores = background_scores
        self.length = len(scores)
        self.zspacing = zspacing
        self.smoothing_sigma = smoothing_sigma
        self.transform_min = transform_min
        self.transform_max = transform_max
        self.tangential_slope_min = tangential_slope_min
        self.tangential_slope_max = tangential_slope_max
        self.slope_mean = slope_mean
        self.scale = 100
        self.original_values = scores
        self.original_transformed_values = self.transform_scores(scores.copy())
        self.a_original, self.b_original = self.fit_linear_line(
            x=np.arange(len(self.original_transformed_values)),
            y=self.original_transformed_values,
        )

        self.values = self.filter_scores(scores)
        self.values = self.transform_scores(self.values)
        self.values = self.smooth_scores(self.values)
        self.set_boundary_indices(self.values)
        self.values = self.remove_outliers(self.values)
        self.valid_region = np.where(~np.isnan(self.values))[0]

        self.z = np.arange(len(scores)) * zspacing
        self.valid_z = self.z[~np.isnan(self.values)]
        self.valid_values = self.values[~np.isnan(self.values)]
        self.a, self.b = self.fit_linear_line(x=self.valid_z, y=self.valid_values)

        # data sanity chekcs
        self.r_slope_threshold = r_slope_threshold
        self.expected_zspacing = self.calculate_expected_zspacing()
        self.r_slope = self.calculate_relative_error_to_expected_slope()
        self.valid_zspacing = self.is_zspacing_valid()
        self.reverse_zordering = self.is_zordering_reverse()

        # define settings
        self.settings = {
            "transform_min": self.transform_min,
            "transform_max": self.transform_max,
            "slope_mean": self.slope_mean,
            "tangential_slope_min": self.tangential_slope_min,
            "tangential_slope_max": self.tangential_slope_max,
            "r_slope_threshold": self.r_slope_threshold,
            "smoothing_sigma": self.smoothing_sigma,
            "background_scores": self.background_scores,
        }

    def __len__(self):
        return len(self.original_values)

    def smooth_scores(self, x):
        smoothed = x.copy()
        not_nan = np.where(~np.isnan(x))
        not_nan_values = x[not_nan]

        # smooth scores
        smoothed[not_nan] = gaussian_filter(
            not_nan_values, sigma=self.smoothing_sigma / self.zspacing
        )

        return np.array(smoothed)

    def transform_scores(self, x):
        if (not np.isnan(self.transform_min)) & (not np.isnan(self.transform_max)):
            transformed = linear_transform(
                x,
                scale=self.scale,
                min_value=self.transform_min,
                max_value=self.transform_max,
            )
            return transformed

    def filter_scores(self, x):
        """Filter predictions of empty slices."""
        for target_score in self.background_scores:
            x[np.round(x, 2) == target_score] = np.nan
        return x

    def set_boundary_indices(self, x):
        min_score = 0
        max_score = 100

        min_boundary_idx = np.nan
        max_boundary_idx = np.nan

        diff_min_score = np.abs(x - min_score)
        diff_max_score = np.abs(x - max_score)

        if np.nanmin(diff_min_score) < 10:
            min_boundary_idx = np.nanargmin(diff_min_score)

        if np.nanmin(diff_max_score) < 10:
            max_boundary_idx = np.nanargmin(diff_max_score)

        self.min_boundary_idx = min_boundary_idx
        self.max_boundary_idx = max_boundary_idx

        if self.min_boundary_idx > self.max_boundary_idx:
            self.min_boundary_idx = np.nan

    def remove_outliers(self, x):
        if len(x) < 2:
            return x

        # get differences. Estimate last difference by copieng previous value
        diffs = np.array(list(np.diff(x)) + [np.diff(x)[-1]])
        self.slopes = diffs / self.zspacing

        # get outlier slopes
        outlier_indices = np.where(
            (self.slopes < self.tangential_slope_min)
            | (self.slopes > self.tangential_slope_max)
        )[0]

        # if unprocessed slice scores increase
        if self.a_original > 0:
            # identify if outliers lie before or after boundary index
            outlier_indices_left_tail = outlier_indices[
                np.where(outlier_indices < self.min_boundary_idx)[0]
            ]
            outlier_indices_right_tail = outlier_indices[
                np.where(outlier_indices > self.max_boundary_idx)[0]
            ]

        # if unprocessed slice scores decrease
        else:
            # identify if outliers lie before or after boundary index
            outlier_indices_left_tail = outlier_indices[
                np.where(outlier_indices < self.max_boundary_idx)[0]
            ]
            outlier_indices_right_tail = outlier_indices[
                np.where(outlier_indices > self.min_boundary_idx)[0]
            ]

        # set left and right tail with outlier slopes to nan
        if len(outlier_indices_left_tail) > 0:
            x[: np.max(outlier_indices_left_tail)] = np.nan
        if len(outlier_indices_right_tail) > 0:
            x[np.min(outlier_indices_right_tail) :] = np.nan
        return x

    def fit_linear_line(self, x, y):
        if len(x) < 2:
            return np.nan, np.nan
        X = np.full((len(x), 2), 1.0, dtype=float)
        X[:, 1] = x
        b, a = np.linalg.inv(X.T @ X) @ X.T @ y
        return a, b

    def calculate_expected_zspacing(self):
        slope_score2index = self.a * self.zspacing
        expected_zspacing = slope_score2index / self.slope_mean

        return expected_zspacing

    def calculate_relative_error_to_expected_slope(self):
        return np.abs(self.a) / self.slope_mean

    def is_zordering_reverse(self):
        if self.a < 0:
            return 1
        return 0

    def is_zspacing_valid(self):
        if np.isnan(self.r_slope):
            return np.nan

        if np.abs(1 - self.r_slope) > self.r_slope_threshold:
            return 0
        return 1
