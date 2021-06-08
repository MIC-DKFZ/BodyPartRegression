import numpy as np 
import pandas as pd
import random, math, cv2
import datetime, os, sys
import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import Dataset
cv2.setNumThreads(1)

class BaseDataset(Dataset):

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
        
        self.data_path = data_path
        self.filenames = filenames
        self.filepaths = [data_path + f for f in filenames]
        self.z_spacings = z_spacings # in mm
        self.length = len(self.filepaths)
        self.num_slices = num_slices
        self.equidistance_range = equidistance_range
        self.custom_transform = custom_transform
        self.random_seed = random_seed
        random.seed(random_seed)

        # define landmark related 
        self.landmark_df = pd.read_excel(landmark_path, sheet_name=landmark_sheet_name, engine='openpyxl', index_col="filename")
        self.landmark_matrix = np.array(self.landmark_df)
        self.landmark_names = self.landmark_df.columns
        self.landmark_files = [f + ".npy" for f in self.landmark_df.index]
        self.landmark_ids = [np.where(f == filenames)[0][0] for f in self.landmark_files]
        self.landmark_slices_per_volume, self.defined_landmarks_per_volume = self.get_landmark_slices()
    
        # define augmentations 
        if custom_transform: 
            self.custom_transform = custom_transform
        
        # Use identity function, if no custom transformation is defined
        else: self.custom_transform = lambda x: x
        
        if albumentation_transform: 
            self.albumentation_transform = albumentation_transform 
        
        # Use identity function, if no transformation is defined 
        else: self.albumentation_transform =  A.Compose([A.Transpose(p=0)])
            
    def _swap_axis(self, x): 
        return x.swapaxes(2, 1).swapaxes(1, 0)
        
    def __len__(self):
        return self.length

    def get_slices(self, filename, indices): 
        volume = np.load(self.data_path + filename, mmap_mode='r')
        x = volume[:, :, indices]
        return self._swap_axis(x)

    def get_full_volume(self, idx: int): 
        filepath = self.filepaths[idx]
        volume = np.load(filepath)
        return self._swap_axis(volume)

    def get_full_volume_from_filepath(self, filepath:str): 
        volume = np.load(filepath)
        return self._swap_axis(volume)

    def get_landmark_idx(self, idx: int): 
        filename = self.filenames[idx]
        idx =np.where(filename == np.array(self.landmark_files))[0]
        if len(idx) == 1: 
            return idx[0]
        return np.nan

    def get_landmark_slices(self): 
        landmark_slices_per_volume =  []
        defined_landmarks_per_volume = []

        for i, file in enumerate(self.landmark_files): 
            # get slice indices of defined landmarks for file
            landmark_indices = self.landmark_matrix[i, :]

            # get information about which landmarks are defined e.g. 0, 1, 2, 3
            defined_landmarks = np.where(~np.isnan(landmark_indices))[0]

            # only save indices of defined landmarks to drop missing values and convert to int
            landmark_indices = landmark_indices[defined_landmarks].astype(int)

            # extract slices of landmark positions for file 
            slices = self.get_slices(file, landmark_indices)

            landmark_slices_per_volume.append(slices)
            defined_landmarks_per_volume.append(defined_landmarks)

        return landmark_slices_per_volume, defined_landmarks_per_volume
