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
    
    def nrrd_header_to_nifti_geometry(self,header):
        """
        Converts NRRD header orientation (usually LPS) to NIfTI geometry (RAS) and extracts spacing & rotation.
        Only supports 'left-posterior-superior' (LPS) and 'right-anterior-superior' (RAS).
        """     
        # Read orientation from header   
        orientation = header.get("space", "").lower()

        # Select transformation matrix from NRRD orientation to RAS
        if orientation == "left-posterior-superior":
            orient_to_ras = np.diag([-1, -1, 1])
        elif orientation == "right-anterior-superior":
            orient_to_ras = np.eye(3)
        else:
            raise ValueError(f"Unsupported orientation: {orientation}. Only LPS and RAS are supported.")

        # Read direction and origin from NRRD header
        directions = np.array(header['space directions'])
        origin = np.array(header['space origin'])

        # Apply orientation transform to directions and transpose for NIfTI convention
        directions_ras = (directions @ orient_to_ras).T

        # Build 4x4 affine: rotation+scaling in upper 3x3
        affine = np.eye(4)
        affine[:3, :3] = directions_ras
        affine[:3, 3] = origin @ orient_to_ras
        
        # Extract voxel spacing (norm of each direction vector) and rotation
        spacing = np.linalg.norm(directions_ras, axis=0)
        rotation = affine[:3, :3]

        return  spacing.astype(np.float32), rotation

    def load_volume(self, filepath):
        try:
            data, header = nrrd.read(filepath)
            data_f = data.astype(np.float32)
        except EOFError:
            print(f"WARNING: Corrupted file {filepath}")
            return None, None
        
        # Get voxel sizes and affine matrix
        pixel_spacings, affine = self.nrrd_header_to_nifti_geometry(header)

        # If affine matrix contains nan's, volume can't be reordered
        try:
            x, pixel_spacings = self.n2n.reorder_volume(
                data_f, pixel_spacings, affine, filepath.split("/")[-1]
            )
        except:
            x, pixel_spacings = np.nan,np.nan

        return x, pixel_spacings