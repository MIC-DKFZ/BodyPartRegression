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


cv2.setNumThreads(1)

sys.path.append("../../")
from bpreg.dataset.base_dataset import BaseDataset, swap_axis


class SSBRDataset(BaseDataset):
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
        indices = self.get_random_slice_indices(volume.shape[2])
        x = volume[:, :, indices]

        # transform each slice seperatly
        for i in range(x.shape[2]):
            x[:, :, i] = self.custom_transform(x[:, :, i])
            x[:, :, i] = self.albumentation_transform(image=x[:, :, i])["image"]

        return swap_axis(x), indices, np.nan

    def get_random_slice_indices(self, z_slices):
        # choose distance k between slices
        max_sampled_slices_for_volume = z_slices // self.num_slices
        k = random.randint(
            self.equidistance_range[0],
            min(self.equidistance_range[1], max_sampled_slices_for_volume),
        )

        slice_range = self.num_slices * k
        min_starting_slice = 0
        max_starting_slice = z_slices - slice_range

        # randomly choose the start slice
        starting_slice = random.randint(min_starting_slice, max_starting_slice)

        slice_indices = np.arange(
            starting_slice, starting_slice + self.num_slices * k, k
        )
        return slice_indices
