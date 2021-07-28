import numpy as np
import pandas as pd
import SimpleITK as sitk
import os, sys
import pickle
from tqdm import tqdm

sys.path.append("../../")
from bpreg.preprocessing.nifti2npy import Nifti2Npy
from scripts.create_config import get_basic_config


def dicom2nifti(ifilepath, ofilepath):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(ifilepath)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, ofilepath)


def convert_ct_lymph_nodes_to_nifti(dicom_path, nifti_path):
    def get_ct_lymph_node_dicom_dir(base_path):
        path = base_path + "/" + os.listdir(base_path)[0] + "/"
        path += os.listdir(path)[0] + "/"
        return path

    dicom_dirs = [
        mydir
        for mydir in np.sort(os.listdir(dicom_path))
        if mydir.startswith(("ABD", "MED"))
    ]

    for dicom_dir in tqdm(dicom_dirs):
        ifilepath = get_ct_lymph_node_dicom_dir(dicom_path + dicom_dir)
        ofilepath = nifti_path + dicom_dir + ".nii.gz"
        if os.path.exists(ofilepath):
            continue
        dicom2nifti(ifilepath, ofilepath)


def nifti2npy(nifti_path, npy_path):
    base_path = "/".join(nifti_path.split("/")[0:-2]) + "/"
    n2n = Nifti2Npy(
        target_pixel_spacing=7,  # in mm/pixel
        min_hu=-1000,  # in Hounsfield units
        max_hu=1500,  # in Hounsfield units
        size=64,  # x/y size
        ipath=nifti_path,  # input path
        opath=npy_path,  # output path
        rescale_max=1,  # rescale max value
        rescale_min=-1,
    )  # rescale min value
    filepaths = np.sort([nifti_path + f for f in os.listdir(nifti_path)])

    df = n2n.convert(filepaths, save=True)
    df.to_excel(base_path + "meta_data.xlsx")


def update_meta_data(landmark_filepath, meta_data_filepath):
    """
    add information to train, val and test data to dataframe
    """
    df_landmarks = pd.read_excel(landmark_filepath, sheet_name="database")
    df_meta_data = pd.read_excel(meta_data_filepath, index_col=0)

    train_filenames = [
        f + ".npy" for f in df_landmarks.loc[df_landmarks.train == 1, "filename"]
    ]
    val_filenames = [
        f + ".npy" for f in df_landmarks.loc[df_landmarks.val == 1, "filename"]
    ]
    test_filenames = [
        f + ".npy" for f in df_landmarks.loc[df_landmarks.test == 1, "filename"]
    ]

    df_meta_data["train_data"] = 1
    df_meta_data.loc[val_filenames, "val_data"] = 1
    df_meta_data.loc[test_filenames, "test_data"] = 1
    df_meta_data.loc[val_filenames + test_filenames, "train_data"] = 0

    df_meta_data.to_excel(meta_data_filepath)


def create_standard_config(base_path, npy_path, config_path, model_path):
    config = get_basic_config(size=64)

    config["df_data_source_path"] = base_path + "meta_data.xlsx"
    config["landmark_path"] = "data/ct-lymph-nodes-annotated-landmarks.xlsx"
    config["data_path"] = npy_path
    config["save_dir"] = model_path
    config["model_name"] = "standard_model"

    with open(config_path + config["name"], "wb") as f:
        pickle.dump(config, f)

    return config


def preprocess_ct_lymph_node_dataset(dicom_path, nifti_path, npy_path): 
    """Convert DICOM files form CT Lymph node to downsampled npy volumes. 
    """
    # Convert Dicom to nifti
    convert_ct_lymph_nodes_to_nifti(dicom_path, nifti_path)
    
    # Convert nifti files to npy and save meta_data.xlsx file 
    nifti2npy(nifti_path, npy_path)
    
    # update meta data with train/val/test data from landmark file 
    update_meta_data(landmark_filepath="data/ct-lymph-nodes-annotated-landmarks.xlsx", 
                     meta_data_filepath="data/meta_data.xlsx")

