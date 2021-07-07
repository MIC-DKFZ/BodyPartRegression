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
import pandas as pd
import random, math, cv2
import datetime, os, sys
import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import Dataset

cv2.setNumThreads(1)

sys.path.append("../../")
from bpreg.dataset.base_dataset import BaseDataset, swap_axis


class BPRDataset(BaseDataset):
    def __init__(
        self,
        data_path: str = "",
        filenames: list = [],
        z_spacings: list = [],
        landmark_path: str = "",
        landmark_sheet_name: str = "",
        num_slices: int = 8,
        equidistance_range: list = [2, 10],
        custom_transform=False,
        albumentation_transform=False,
        random_seed: int = 0,
    ):

        BaseDataset.__init__(
            self,
            data_path,
            filenames,
            z_spacings,
            landmark_path,
            landmark_sheet_name,
            num_slices,
            equidistance_range,
            custom_transform,
            albumentation_transform,
            random_seed,
        )

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        z_spacing = self.z_spacings[idx]
        volume = np.load(filepath, mmap_mode="r")
        indices, physical_distance = self.get_random_slice_indices(
            volume.shape[2], z_spacing
        )
        x = volume[:, :, indices]

        # transform each slice seperatly
        for i in range(x.shape[2]):
            x[:, :, i] = self.custom_transform(x[:, :, i])
            x[:, :, i] = self.albumentation_transform(image=x[:, :, i])["image"]

        return swap_axis(x), indices, physical_distance

    def get_random_slice_indices(self, z, z_spacing):
        # set z-distance sampling range in terms of slices in between
        # round first value up and second value down
        equidistance_range = [
            math.ceil(self.equidistance_range[0] / z_spacing),
            math.floor(self.equidistance_range[1] / z_spacing),
        ]

        # randomly select a slice difference k
        if z <= max(equidistance_range) * (self.num_slices - 1):
            max_dist = math.floor((z - 1) / (self.num_slices - 1))
            k = random.randint(equidistance_range[0], max_dist)
        else:
            k = random.randint(*equidistance_range)
        physical_distance = k * z_spacing
        myRange = (self.num_slices - 1) * k
        min_starting_slice = 0
        max_starting_slice = (z - 1) - myRange
        starting_slice = random.randint(min_starting_slice, max_starting_slice)
        slice_indices = np.arange(
            starting_slice, starting_slice + self.num_slices * k, k
        )
        return slice_indices, np.array([physical_distance] * (self.num_slices - 1))
