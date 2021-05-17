import numpy as np 
import pandas as pd
import random, math, cv2
import datetime, os, sys
import pytorch_lightning as pl
import albumentations as A
from torch.utils.data import Dataset
cv2.setNumThreads(1)

class BPRDataset(Dataset):

    def __init__(self, 
                 filepaths, 
                 z_spacings,
                 landmark_path=None, 
                 landmark_sheet_name=False,
                 num_slices=8, 
                 equidistance_range=[2, 10], 
                 delta_z_max=np.inf,
                 custom_transform=False, 
                 albumentation_transform=False, 
                 random_seed=0,
                 drop_landmarks=[]):
        
        self.filepaths = filepaths
        self.z_spacings = z_spacings # in mm
        self.filenames = np.array([f.split("/")[-1] for f in filepaths])
        self.length = len(self.filepaths)
        self.num_slices = num_slices
        self.equidistance_range = equidistance_range
        self.custom_transform = custom_transform
        if landmark_path:
            self.landmarks = self.get_landmarks_dict(landmark_path, landmark_sheet_name, drop_landmarks=drop_landmarks)
    
        if custom_transform: 
            self.custom_transform = custom_transform
        
        # Use identity function, if no custom transformation is defined
        else: self.custom_transform = lambda x: x
        
        if albumentation_transform: 
            self.albumentation_transform = albumentation_transform 
        
        # Use identity function, if no transformation is defined 
        else: self.albumentation_transform =  A.Compose([A.Transpose(p=0)])
            
        self.random_seed = random_seed
        random.seed(random_seed)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        z_spacing = self.z_spacings[idx]
        volume = np.load(filepath, mmap_mode='r')
        indices, physical_distance  = self.get_random_slice_indices(volume.shape[2], z_spacing)
        x = volume[:, :, indices]
        
        # transform each slice seperatly 
        for i in range(x.shape[2]): 
            x[:, :, i] = self.custom_transform(x[:, :, i])
            x[:, :, i] = self.albumentation_transform(image=x[:, :, i])["image"]
                
        return x.swapaxes(2, 1).swapaxes(1, 0), indices, physical_distance
    
    def get_random_slice_indices(self, z, z_spacing):
        
        # convert equidistance range in slice index difference s
        # round first value up and second value down  
        equidistance_range = [math.ceil(self.equidistance_range[0]/z_spacing), 
                              math.floor(self.equidistance_range[1]/z_spacing)]
        
        if z <= max(equidistance_range)*(self.num_slices-1): 
            max_dist = math.floor((z-1)/(self.num_slices-1))
            dist = random.randint(equidistance_range[0], max_dist)
        else: 
            dist = random.randint(*equidistance_range)
        physical_distance = dist * z_spacing
        myRange = (self.num_slices-1) * dist
        min_starting_slice  = 0
        max_starting_slice = (z-1) - myRange 
        starting_slice = random.randint(min_starting_slice, max_starting_slice)
        slice_indices = np.arange(starting_slice, starting_slice + self.num_slices * dist, dist)
        return slice_indices, np.array([physical_distance]* (self.num_slices - 1))

    # @multimethod()   # TODO 
    def get_full_volume(self, idx: int): 
        filepath = self.filepaths[idx]
        volume = np.load(filepath)
        return volume.swapaxes(2, 1).swapaxes(1, 0)
    
    
    def get_landmarks_dict(self, path, sheet_name=False, drop_landmarks=[]):
        def isnan(x): 
            return not (x==x)

        if sheet_name: dfl = pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')
        else: dfl = pd.read_excel(path, engine='openpyxl')
        dfl.drop(drop_landmarks, axis=1, inplace=True)
        if "Unnamed: 0" in dfl.columns: dfl.drop("Unnamed: 0" , axis=1, inplace=True)
        dfl = dfl.dropna(how="all")
        landmark_filenames = dfl.filename.values + ".npy"

        landmarks = {key: {} for key in range(len(dfl))}

        self.landmark_names = [col for col in dfl.columns if col != "filename"]
        for i, f in enumerate(landmark_filenames): 
            # fix autocorrectur of CT-COLON files 
            f = f.replace("—","--")
            landmarks[i]["filename"] = f
            
            # get index of landmark in current dataset
            try: index = np.where(self.filenames == f)[0][0]
            except: 
                print(f, np.where(self.filenames == f)) 
                raise ValueError(f"Filename {f} from landmark-file {path} not found.")
           
            landmarks[i]["dataset_index"] = index
            
            defined_landmarks = np.full(len(self.landmark_names), np.nan)
            landmark_slice_indices = []

            for j, col in enumerate(self.landmark_names): 
                if not (isnan(dfl.loc[i, col])): 
                    defined_landmarks[j] = 1 
                    landmark_slice_indices.append(int(dfl.loc[i, col]))

            landmarks[i]["slice_indices"] = landmark_slice_indices
            landmarks[i]["defined_landmarks"] = defined_landmarks
            landmarks[i]["defined_landmarks_i"] = np.where(landmarks[i]["defined_landmarks"] == 1)[0]

        return landmarks
