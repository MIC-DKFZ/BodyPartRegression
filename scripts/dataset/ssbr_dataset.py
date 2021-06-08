import numpy as np 
import pandas as pd
import random, math, cv2
import datetime, os, sys
import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import Dataset
cv2.setNumThreads(1)

sys.path.append("../../")
from scripts.dataset.base_dataset import BaseDataset

class SSBRDataset(BaseDataset):

    def __init__(self, 
                 data_path="", 
                 filenames=[],
                 z_spacings=[],
                 landmark_path=None, 
                 landmark_sheet_name=False,
                 num_slices=8, 
                 equidistance_range=[2, 10], 
                 delta_z_max=np.inf,
                 custom_transform=False, 
                 albumentation_transform=False, 
                 random_seed=0,
                 drop_landmarks=[]):
        
        BaseDataset.__init__(self, data_path, filenames, z_spacings, landmark_path, 
                             landmark_sheet_name, num_slices, equidistance_range, 
                             delta_z_max, custom_transform, albumentation_transform, 
                             random_seed, drop_landmarks)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        z_spacing = self.z_spacings[idx]
        volume = np.load(filepath, mmap_mode='r')
        indices  = self.get_random_slice_indices(volume.shape[2])
        x = volume[:, :, indices]
        
        # transform each slice seperatly 
        for i in range(x.shape[2]): 
            x[:, :, i] = self.custom_transform(x[:, :, i])
            x[:, :, i] = self.albumentation_transform(image=x[:, :, i])["image"]
                
        return self._swap_axis(x), indices, np.nan

    def get_random_slice_indices(self, z_slices): 
        # choose distance k between slices
        max_sampled_slices_for_volume = z_slices//self.num_slices
        k = random.randint(self.equidistance_range[0], min(self.equidistance_range[1], max_sampled_slices_for_volume))

        slice_range = self.num_slices * k
        min_starting_slice  = 0
        max_starting_slice = z_slices - slice_range

        # randomly choose the start slice
        starting_slice = random.randint(min_starting_slice, max_starting_slice)
        
        slice_indices = np.arange(starting_slice, starting_slice + self.num_slices * k, k)
        return slice_indices