import numpy as np
import random
import cv2


class rescale_intensity(object):
    def __init__(self, low, high, scale=255, dtype=np.uint8):
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
        alpha_min=0,
        alpha_max=1,
        beta_min=1.5,
        beta_max=3.5,
        p_alpha=0.75,
        p_beta=0.75,
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
        self, r_square=0.6, r_circle=0.75, dimension=128, p=0.2, fill_value=-1.0
    ):
        self.p = p
        self.d = dimension
        self.r_square = r_square
        self.r_circle = r_circle
        self.fill_value = fill_value
        self.square_frame = self.get_square_frame()
        self.circle_frame = self.get_circle_frame()

    def get_square_frame(self):
        irange = int(self.d * self.r_square + 0.5)
        if irange % 2 == 0:
            irange += 1

        X = np.full((self.d, self.d), np.nan)
        X_inside = np.full((irange, irange), 0)
        X[
            self.d // 2 - irange // 2 : self.d // 2 + irange // 2 + 1,
            self.d // 2 - irange // 2 : self.d // 2 + irange // 2 + 1,
        ] = X_inside

        return X  # [:, :, np.newaxis]

    def get_circle_frame(self):
        irange = int(self.d * self.r_circle + 0.5)
        if irange % 2 == 0:
            irange += 1
        X = np.full((self.d, self.d), np.nan)
        X_inside = np.full((irange, irange), 0, dtype=float)
        center = irange // 2
        radius = center
        for idx in range(irange):
            for idy in range(irange):

                if (idx - center) ** 2 + (idy - center) ** 2 > radius ** 2:
                    X_inside[idx, idy] = np.nan

        X[
            self.d // 2 - irange // 2 : self.d // 2 + irange // 2 + 1,
            self.d // 2 - irange // 2 : self.d // 2 + irange // 2 + 1,
        ] = X_inside

        return X  # [:, :, np.newaxis]

    def __call__(self, x):
        if random.random() < self.p:
            x = x + self.circle_frame
            x[np.isnan(x)] = self.fill_value
            x = np.nan_to_num(x)
        return x


class GaussNoise(object):
    def __init__(self, std_min=0, std_max=5, min_value=0, max_value=255, p=0.5):
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
    def __init__(self, shift_limit=2.55, max_value=255, min_value=0, p=0.5):
        self.limit = shift_limit
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
    def __init__(self, scale_delta=0.2, min_value=0, max_value=255, p=1):
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
    def __init__(self, gamma_min, gamma_max, max_value=255, p=0.5):
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
