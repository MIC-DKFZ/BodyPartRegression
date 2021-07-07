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
import random
import cv2


class rescale_intensity(object):
    def __init__(
        self, low: float, high: float, scale: float = 2, dtype: type = np.uint8
    ):
        self.low = low
        self.high = high
        self.scale = scale
        self.dtype = dtype

    def __call__(self, x):
        x[x < self.low] = self.low
        x[x > self.high] = self.high
        x = x - self.low
        x = x / (self.high - self.low)
        x = x * self.scale
        x = x.astype(dtype=self.dtype)

        return x


class adjust_contrast(object):
    def __init__(
        self,
        alpha_min: float = 0,
        alpha_max: float = 1,
        beta_min: float = 1.5,
        beta_max: float = 3.5,
        p_alpha: float = 0.75,
        p_beta: float = 0.75,
    ):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_diff = alpha_max - alpha_min

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_diff = beta_max - beta_min

        self.p_alpha = p_alpha
        self.p_beta = p_beta

    def __call__(self, x):
        mean = np.mean(x)
        std = np.std(x)

        if random.random() <= self.p_alpha:
            a_rand = random.random() * self.alpha_diff + self.alpha_min
            min_value = mean - a_rand * std
            x = np.where(x < min_value, min_value, x)

        if random.random() <= self.p_beta:
            b_rand = random.random() * self.beta_diff + self.beta_min
            max_value = mean + b_rand * std
            x = np.where(x > max_value, max_value, x)

        return x


class AddFrame:
    def __init__(
        self,
        r_square: float = 0.6,
        r_circle: float = 0.75,
        dimension: int = 128,
        p: float = 0.2,
        fill_value: float = -1.0,
    ):
        self.p = p
        self.dimension = dimension
        self.r_square = r_square
        self.r_circle = r_circle
        self.fill_value = fill_value
        self.square_frame = self.get_square_frame()
        self.circle_frame = self.get_circle_frame()

    def get_square_frame(self):
        irange = int(self.dimension * self.r_square + 0.5)
        if irange % 2 == 0:
            irange += 1

        X = np.full((self.dimension, self.dimension), np.nan)
        X_inside = np.full((irange, irange), 0)
        X[
            self.dimension // 2 - irange // 2 : self.dimension // 2 + irange // 2 + 1,
            self.dimension // 2 - irange // 2 : self.dimension // 2 + irange // 2 + 1,
        ] = X_inside

        return X  # [:, :, np.newaxis]

    def get_circle_frame(self):
        irange = int(self.dimension * self.r_circle + 0.5)
        if irange % 2 == 0:
            irange += 1
        X = np.full((self.dimension, self.dimension), np.nan)
        X_inside = np.full((irange, irange), 0, dtype=float)
        center = irange // 2
        radius = center
        for idx in range(irange):
            for idy in range(irange):

                if (idx - center) ** 2 + (idy - center) ** 2 > radius ** 2:
                    X_inside[idx, idy] = np.nan

        X[
            self.dimension // 2 - irange // 2 : self.dimension // 2 + irange // 2 + 1,
            self.dimension // 2 - irange // 2 : self.dimension // 2 + irange // 2 + 1,
        ] = X_inside

        return X  # [:, :, np.newaxis]

    def __call__(self, x):
        if random.random() < self.p:
            x = x + self.circle_frame
            x[np.isnan(x)] = self.fill_value
            x = np.nan_to_num(x)
        return x


class GaussNoise(object):
    def __init__(
        self,
        std_min: float = 0,
        std_max: float = 5,
        min_value: float = -1,
        max_value: float = 1,
        p: float = 0.5,
    ):
        self.std_min = std_min
        self.std_max = std_max
        self.min_value = min_value
        self.max_value = max_value
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            sigma = random.uniform(self.std_min, self.std_max)
            gauss = np.random.normal(0, sigma, x.shape)
            x = x + gauss
            x[x < self.min_value] = self.min_value
            x[x > self.max_value] = self.max_value
        return x


class ShiftHU(object):
    def __init__(
        self,
        limit: float = 2.55,
        max_value: float = 1,
        min_value: float = -1,
        p: float = 0.5,
    ):
        self.limit = limit
        self.p = p
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x):
        if random.random() < self.p:
            shift = random.random() * 2 * self.limit - self.limit
            x = np.where(x > 0, x + shift, x)
            x[x < self.min_value] = self.min_value
            x[x > self.max_value] = self.max_value
        return x


class ScaleHU:
    def __init__(
        self,
        scale_delta: float = 0.2,
        min_value: float = -1,
        max_value: float = 1,
        p: float = 1,
    ):
        self.p = p
        self.scale_delta = scale_delta
        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, x):
        if random.random() < self.p:
            factor = np.random.uniform(1 - self.scale_delta, 1 + self.scale_delta)
            x = x * factor
            x[x > self.max_value] = self.max_value
            x[x < self.min_value] = self.min_value

        return x


class RandomGamma:
    def __init__(
        self, gamma_min: float, gamma_max: float, max_value: float = 1, p: float = 0.5
    ):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.max_value = max_value
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            gamma = np.random.uniform(self.gamma_min, self.gamma_max)
            x = np.power(x, gamma)
            x[x > self.max_value] = self.max_value
            x = x.astype(np.float32)
        return x
