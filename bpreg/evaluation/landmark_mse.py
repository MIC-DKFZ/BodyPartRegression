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


class LMSE:
    """class to calculate the landmark mean square error.
    The landmark mean square error is a normalized verison of the mean square error.
    As normalization constant d the difference between the expected slice scores of two landmarks is used.
    """

    def __init__(self):
        pass

    def from_dataset(self, model, val_dataset, train_dataset):
        val_score_matrix = model.compute_slice_score_matrix(val_dataset)
        train_score_matrix = model.compute_slice_score_matrix(train_dataset)
        d = self.get_normalizing_constant(np.nanmean(train_score_matrix, axis=0))
        mse, mse_std = self.from_matrices(val_score_matrix, train_score_matrix)

        return mse, mse_std, d

    def lmse_per_landmark_from_matrices(self, score_matrix, reference_matrix, d=False):
        square_error_matrix = self.get_square_error_matrix(
            score_matrix, reference_matrix, d=d
        )
        lmse_per_landmark = np.nanmean(square_error_matrix, axis=0)

        counts = np.sum(np.where(~np.isnan(square_error_matrix), 1, 0), axis=0)
        lmse_errors = np.nanstd(square_error_matrix, ddof=1, axis=0) / np.sqrt(counts)

        return lmse_per_landmark, lmse_errors

    def lmse_per_volume_from_matrices(self, score_matrix, reference_matrix, d=False):
        square_error_matrix = self.get_square_error_matrix(
            score_matrix, reference_matrix, d=d
        )
        lmse_per_volume = np.nanmean(square_error_matrix, axis=1)
        counts = np.sum(np.where(~np.isnan(square_error_matrix), 1, 0), axis=1)
        lmse_errors = np.nanstd(square_error_matrix, ddof=1, axis=1) / np.sqrt(counts)

        return lmse_per_volume, lmse_errors

    def lmse_per_slice_from_matrices(self, score_matrix, reference_matrix, d=False):
        square_error_matrix = self.get_square_error_matrix(
            score_matrix, reference_matrix, d=d
        )
        lmse = np.nanmean(square_error_matrix)
        counts = np.sum(np.where(~np.isnan(square_error_matrix), 1, 0))
        lmse_std = np.nanstd(square_error_matrix, ddof=1) / np.sqrt(counts)

        return lmse, lmse_std

    def from_matrices(self, score_matrix, reference_matrix, d=False):
        lmses, lmse_stds = self.lmse_per_volume_from_matrices(
            score_matrix, reference_matrix, d=d
        )
        return np.mean(lmses), np.std(lmses, ddof=1) / np.sqrt(len(lmses))

    def get_square_error_matrix(self, score_matrix, reference_matrix, d=False):
        expected_scores = np.nanmean(reference_matrix, axis=0)
        if not d:
            d = np.abs(self.get_normalizing_constant(expected_scores))
        square_error_matrix = self.from_instance(expected_scores, score_matrix, d)
        return square_error_matrix

    def from_volume(self, landmarks, scores, expected_scores):
        d = self.get_normalizing_constant(expected_scores)
        expected_scores = expected_scores[~np.isnan(landmarks)]
        landmarks = np.array(landmarks[~np.isnan(landmarks)]).astype(int)
        scores = scores[landmarks]
        return self.from_instance(scores, expected_scores, d)

    def from_instance(self, y_truth, y_pred, d):
        mse = ((y_truth - y_pred) / d) ** 2
        return mse

    def get_normalizing_constant(self, expected_scores):
        return (expected_scores[-1] - expected_scores[0]) / 100
