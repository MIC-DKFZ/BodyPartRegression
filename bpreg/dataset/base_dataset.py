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
import os
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import random, math, cv2
import albumentations as A
from torch.utils.data import Dataset

cv2.setNumThreads(1)


class BaseDataset(Dataset):
    """Parent class to create a dataset for a body part regression model.

    Args:
        data_path (str, optional): Path where .npy data is stored. Defaults to "".
        filenames (list, optional): filenames of .npy volumes. Defaults to [].
        z_spacings (list, optional): corresponding z-spacings to .npy volumes. Defaults to [].
        landmark_path (str, optional): path to landmark-file which stores the annotated landmarks for a
        subset of the training data, for the validation data and for the test data. Defaults to None.
        landmark_sheet_name (str, optional): sheet-name of landmark file which corresponds to dataset e.g. val-landmarks . Defaults to "".
        num_slices (int, optional): Numer of sampled slices per volume.
        equidistance_range (list, optional): Distance of sampling num_slices from volume.
        custom_transform (bool, optional): custom slice wise transformations. Defaults to False.
        albumentation_transform (bool, optional): slice wise transformations from albumentation. Defaults to False.
        random_seed (int, optional): define random seed for sampling. Defaults to 0.
    """

    def __init__(
        self,
        data_path: str = "",
        filenames: list = [],
        z_spacings: list = [],
        landmark_path: str = None,
        landmark_sheet_name: str = "",
        num_slices: int = 4,
        equidistance_range: list = [5, 100],
        custom_transform=False,
        albumentation_transform=False,
        random_seed: int = 0,
    ):

        self.data_path = data_path
        self.filenames = filenames
        self.filepaths = [os.path.join(data_path, f) for f in filenames]
        self.z_spacings = z_spacings  # in mm
        self.length = len(self.filepaths)
        self.num_slices = num_slices
        self.equidistance_range = equidistance_range
        self.custom_transform = custom_transform
        self.random_seed = random_seed
        random.seed(random_seed)

        # define landmark related
        if landmark_path:
            self.landmark_df = pd.read_excel(
                landmark_path,
                sheet_name=landmark_sheet_name,
                engine="openpyxl",
                index_col="filename",
            )
            self.landmark_matrix = np.array(self.landmark_df)
            self.landmark_names = self.landmark_df.columns
            self.landmark_files = [
                f + ".npy" for f in self.landmark_df.index if isinstance(f, str)
            ]
            self.landmark_ids = [
                filename_to_id(f, filenames) for f in self.landmark_files
            ]
            (
                self.landmark_slices_per_volume,
                self.defined_landmarks_per_volume,
            ) = self.get_landmark_slices()

        # define augmentations
        if custom_transform:
            self.custom_transform = custom_transform

        # Use identity function, if no custom transformation is defined
        else:
            self.custom_transform = lambda x: x

        if albumentation_transform:
            self.albumentation_transform = albumentation_transform

        # Use identity function, if no transformation is defined
        else:
            self.albumentation_transform = A.Compose([A.Transpose(p=0)])

    def __len__(self):
        return self.length

    def get_full_volume(self, idx: int):
        filepath = self.filepaths[idx]
        volume = np.load(parse2plainname(filepath) + ".npy")
        return swap_axis(volume)

    def get_landmark_idx(self, idx: int):
        filename = self.filenames[idx]
        idx = np.where(filename == np.array(self.landmark_files))[0]
        if len(idx) == 1:
            return idx[0]
        return np.nan

    def get_landmark_slices(self):
        landmark_slices_per_volume = []
        defined_landmarks_per_volume = []

        for i, file in enumerate(self.landmark_files):
            # get slice indices of defined landmarks for file
            landmark_indices = self.landmark_matrix[i, :]

            # get information about which landmarks are defined e.g. 0, 1, 2, 3
            defined_landmarks = np.where(~np.isnan(landmark_indices))[0]

            # only save indices of defined landmarks to drop missing values and convert to int
            landmark_indices = landmark_indices[defined_landmarks].astype(int)

            # extract slices of landmark positions for file
            slices = get_slices(os.path.join(self.data_path, file), landmark_indices)

            landmark_slices_per_volume.append(slices)
            defined_landmarks_per_volume.append(defined_landmarks)

        return landmark_slices_per_volume, defined_landmarks_per_volume


def get_full_volume_from_filepath(filepath: str):
    volume = np.load(filepath)
    return swap_axis(volume)


def get_slices(filepath, indices):
    volume = np.load(parse2plainname(filepath) + ".npy", mmap_mode="r")
    x = volume[:, :, indices]
    return swap_axis(x)


def swap_axis(x):
    return x.swapaxes(2, 1).swapaxes(1, 0)


def filename_to_id(filename, filename_array):
    filename = parse2plainname(filename)
    filename_array = parse2plainname(filename_array)

    ids = np.where(filename == np.array(filename_array))[0]
    if len(ids) == 0:
        raise ValueError(
            f"filename {filename} is not in the filename list: {filename_array}"
        )
    else:
        return ids[0]


def parse2plainname(value):
    if isinstance(value, str):
        value = value.replace(".nii", "").replace(".gz", "").replace(".npy", "")
    else:
        value = [
            v.replace(".nii", "").replace(".gz", "").replace(".npy", "") for v in value
        ]
    return value
