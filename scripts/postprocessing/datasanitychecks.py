import torch
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import gaussian_filter


class DataSanityCheck:
    """Class checks if z-ordering is reverse or not and 
    if the z-spacing seems to be plausible. 
    Additionaly, it processes the slice score prediction for 
    a CT volume by smoothing it and finding an invalid ranges. 

    Args:
        scores (np.ndarray): unprocessed predicted slice-scores
        zspacing (float): z-spacing of CT volume
        slope_mean (float): mean slice-score slope for used model
        slope_std (float): standard deviation for slice-score slope for used model
        lower_bound_score (float): min expected slice score. If the slice-scores monotonusly 
        decrease before the lower bound score, the area is defined as invalid and the cleaned slice 
        scores are set to none. 
        upper_bound_score (float): max expected slice score. If the slice-scores monotonously decrease
        after the upper bound score, the area is defined as invalid and the cleand slice
        scores are set to none
        smoothing_sigma (float, optional): smoothing sigma in mm for gaussian smoothing for the slice-scores.
        Defaults to 10 mm.
    """
    def __init__(
        self,
        scores: np.ndarray,
        zspacing: float,
        slope_mean: float,
        slope_std: float,
        lower_bound_score: float,
        upper_bound_score: float,
        smoothing_sigma: float = 10,
    ):
        self.scores = scores
        self.slope_mean = slope_mean
        self.slope_std = slope_std
        self.z = zspacing
        self.smoothing_sigma = smoothing_sigma 
        self.lower_bound_score = lower_bound_score
        self.upper_bound_score = upper_bound_score
        if len(scores) == 1: 
            self.a, self.b = np.nan, np.nan 
            self.zhat, self.zhat_std = np.nan, np.nan 
        else: 
            self.a, self.b = self.fit_line(scores, zspacing)

            # get expected z-spacing
            self.zhat, self.zhat_std = self.estimate_zspacing(scores)

    def fit_line(self, scores, zspacing=1):
        # Use robust linear regression
        x = np.arange(0, len(scores)) * zspacing
        X = np.ones((len(scores), 2))
        X[:, 0] = x
        reg = RANSACRegressor(random_state=0).fit(X, scores)
        reg.score(X, scores)
        inliers = (reg.inlier_mask_) * 1
        a, b = reg.estimator_.coef_[0], reg.estimator_.intercept_

        return a, b

    def estimate_zspacing(self, scores):
        a, b = self.fit_line(scores)
        z_hat = a / self.slope_mean
        delta_z_hat = a * self.slope_std / self.slope_mean ** 2

        return z_hat, delta_z_hat

    def is_reverse_zordering(self):
        if self.a < 0:
            return 1
        return 0

    def is_valid_zspacing(self):
        if ((self.zhat - 3 * self.zhat_std) < self.z) and (
            self.z < (self.zhat + 3 * self.zhat_std)
        ):
            return 1
        return 0

    def get_index_bounds(self, negative_slope_indices, scores):
        """
        Get first negative slope index before the lower bound score and
        the first negative slope after the upper bound score.
        With this procedure slice scores outside the valid range (upper head and legs) should be set to nan-values.
        """

        if np.min(scores) > self.lower_bound_score:
            min_valid_index = 0
        else:
            min_valid_index = max(np.where(scores < self.lower_bound_score)[0])
            min_invalid_indices = negative_slope_indices[
                np.where(negative_slope_indices <= min_valid_index)
            ]
            if len(min_invalid_indices) > 0:
                min_valid_index = max(min_invalid_indices)
            else:
                min_valid_index = 0

        upper_scores = scores[min_valid_index:]

        if np.max(scores) < self.upper_bound_score:
            max_valid_index = len(scores)
        else:
            max_valid_index = min(np.where(upper_scores > self.upper_bound_score)[0])
            max_invalid_indices = negative_slope_indices[
                np.where(negative_slope_indices >= max_valid_index)
            ]
            if len(max_invalid_indices) > 0:
                max_valid_index = min(max_invalid_indices)
            else:
                max_valid_index = len(scores)

        return min_valid_index, max_valid_index

    def remove_invalid_regions(self, scores):
        smoothed_scores = gaussian_filter(scores, sigma=self.smoothing_sigma / self.z)

        diff = np.array(smoothed_scores[1:]) - np.array(smoothed_scores[:-1])
        negative_slope_indices = np.where(diff < 0)[0]

        min_valid_index, max_valid_index = self.get_index_bounds(
            negative_slope_indices, smoothed_scores
        )

        valid_indices = np.arange(min_valid_index, max_valid_index + 1)
        cleaned_scores = np.round(smoothed_scores.copy(), 4)
        cleaned_scores[0:min_valid_index] = np.nan
        cleaned_scores[max_valid_index + 1 :] = np.nan

        return cleaned_scores, valid_indices
