import numpy as np
import nibabel as nib
import nrrd

import pandas as pd
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from bpreg.preprocessing.nifti2npy import Nifti2Npy


class Nrrd2Npy:
    """Convert nrrd files to numpy arrays

    Args:
        ipath (str, optional): input path of nifti-files. Defaults to "/home/username/Documents/Data/DataSet/Images/".
    """

    def __init__(
            self,
        ):
            self.n2n = Nifti2Npy(
                target_pixel_spacing=3.5, min_hu=-1000, max_hu=1500, size=128
            )

    def preprocess_nrrd(self, filepath: str):
        x, pixel_spacings = self.load_volume(filepath)
        x = self.n2n.rescale_xy(x)
        x = self.n2n.resize_volume(x, pixel_spacings)
        return x, pixel_spacings

    def load_volume(self, filepath):
        try:
            data, header = nrrd.read(filepath)
            data_f = data.astype(np.float32)
        except EOFError:
            print(f"WARNING: Corrupted file {filepath}")
            return None, None
        
        # get voxel sizes from dic, equivalent to header.get_zooms in nibabel
        pixel_spacings = np.linalg.norm(header["space directions"], axis=0) 
        affine = header.get('space directions')

        # If affine matrix contains nan's, volume can't be reordered
        try:
            x, pixel_spacings = self.n2n.reorder_volume(
                data_f, pixel_spacings, affine, filepath.split("/")[-1]
            )
        except:
            x, pixel_spacings = np.nan,np.nan

        return x, pixel_spacings