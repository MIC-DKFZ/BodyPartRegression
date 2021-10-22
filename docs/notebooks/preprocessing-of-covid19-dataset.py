# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: bpreg5
#     language: python
#     name: bpreg5
# ---

# <img src="images/body-part-regression-title.png" width=1000/>

# +
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import nibabel as nib
import json, sys

from ipywidgets import widgets, interactive
from bpreg.scripts.bpreg_inference import bpreg_inference
from bpreg.preprocessing.nifti2npy import load_nifti_volume, Nifti2Npy
from bpreg.settings.settings import * 
from tqdm import tqdm 

sys.path.append("../")
from utils import * 


# -

def crop_body_part(X, scores, lower_bound, upper_bound): 
    diff = np.abs(scores - lower_bound)
    lower_bound_idx  = np.where(diff == np.nanmin(diff))[0][0]

    diff = np.abs(scores - upper_bound)
    upper_bound_idx  = np.where(diff == np.nanmin(diff))[0][0]

    if lower_bound_idx < upper_bound_idx: 
        idx_array = np.arange(lower_bound_idx, upper_bound_idx)
    else: 
        idx_array = np.arange(upper_bound_idx, lower_bound_idx)

    #if len(scores) == X.shape[1]: X_cropped = X[:, idx_array, :]
    X_cropped = X[:, :, idx_array]
    return X_cropped 


# +
cropped_path = "/home/AD/s429r/Documents/Data/PyData-Global/cropped_nifti_files/"
cropped_json_path = "/home/AD/s429r/Documents/Data/PyData-Global/cropped_output_files/"

def crop_ct_images(nifti_filepaths, 
                   json_path, 
                   cropped_path, 
                   lower_landmark="lung_start", 
                   upper_landmark="lung_end", 
                   plot=False, 
                   save=False, 
                   gpu_available=False): 
    
    n2n = Nifti2Npy()
    for i, filepath in tqdm(enumerate(nifti_filepaths)): 
        file = filepath.split("/")[-1]
        with open(json_path + file.replace(".nii.gz", ".json"), "rb") as f:  
            myDict = json.load(f)
            X, pixel_spacings = n2n.load_volume(filepath)

            # lower and upper bound of CHEST 
            lookup_table = pd.DataFrame(myDict["look-up table"]).T
            upper_bound = np.round(lookup_table.loc[upper_landmark]["mean"], 0) + 1
            lower_bound = np.round(lookup_table.loc[lower_landmark]["mean"], 0)  - 1

            X_cropped = crop_body_part(X, myDict["cleaned slice scores"], lower_bound, upper_bound)
            
            if myDict["reverse z-ordering"]: X_cropped = np.flip(X_cropped, axis=2)

            if plot: 
                plt.plot(myDict["z"], myDict["cleaned slice scores"])
                plt.plot(myDict["z"], myDict["unprocessed slice scores"])
                plt.title(file)
                plt.show()

                plt.imshow(X[:, X.shape[1]//2, :].T, origin="lower", cmap="gray")
                plt.show()
                plt.imshow(X_cropped[:,  X.shape[1]//2, :].T, origin="lower", cmap="gray")
                plt.show()

            if save:
                new_image = nib.Nifti1Image(X_cropped, affine=np.diag(list(pixel_spacings) + [0]) )
                nib.save(new_image, cropped_path + file.replace(".json", ".nii.gz")) 


# -

# # TODO 
#
# 1. Clean up code 
# 2. Get ride of NONE images 
# 3. Add Notebook + helper functions to BodyPartRegression/doc 
# 4. Test presentation 
# 5. Pfade anpassen 
#
# - Bilder & Jupyter notebook -> github 
# - Daten auf USB back-up erstellen 
# - add Body Part prediction to plot function 
# - transfer notebook -> BodyPartRegression/doc 
# - clean jupyter notebook 
# - add image: nii -> json 
# - show example json file in presentation 
# - test locally 
# - change body part regresssion explanation image 
# - test presentation 
# - add image for final thankyou 
# - jupyter notebook fast enough? 
# - Convert jupyter notebook to html 
# - vertebra of the spine image 
# - prepare slideshow for sharing 
# - uploa Jupyter notebook 
# - Change Text/Markdown Style for Notebook 
# - HTML back-up 
# - Provide data to download on Zenodo! 
#     - Subset of CT scans from the COVID-19-AR dataset in the nifti file format for the body part regression tutorial 
#
# # Analyze chest CT scans in COVID-19 dataset
#
#
# ## Preprocessing Steps: 
#
# To analyze the lungs in new dataset: 
# 1. Filter corrupted/invalid CT scans 
# 1. find CT volumes which include lungs
# 2. Crop volumes, so that only the lungs are visible
# ---------------------------
#
# Download data from study **COVID-19-AR** from the TCIA
# - 195 CT scans 
#
# ## TODO: 
#     - Show non chest CT scans 
#     - Show corrupted CT scans 
#     - Show CT scans with more body regions 

# # Body Part Recognition with Python 

# define paths 
nifti_path = "COVID-19-AR/nifti_files/"
json_path = "COVID-19-AR/output_files/"
cropped_path = "COVID-19-AR/cropped_nifti_files/"
cropped_json_path = "COVID-19-AR/cropped_output_files/"


# ## 1. Download Data 
# - Download COVID-19 CT dataset from the TCIA: 
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226443
# - Convert the DICOM files to NIFTI files. I recommend to use the simpleITK package for the conversion. 
# --------------------------- 
# <img src="images/tcia-screenshot.png" width=1000/>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>
# <br>

# ## 2. Analyze Data 

plot_volumes_interactive(nifti_path) 

# <br> 
# <br> 
# <br> 
# <br> 
# <br>
# <br>
# <br>
#
# --------------------------------------------
#
#
# <img src="images/problems-with-covid19-dataset.png" width=1000/>
# <br> 
# <br> 
# <br> 
# <br> 
# <br>
# <br>
# <br>

# ## 3. Define preprocessing steps 
# 1. Remove invalid CT scans 
#     - filter CT scans, with wrong axis ordering
#     - filter corrupted CT scans, where no human body is visible 
#     - filter CT scans with less than 20 slices 
# 2. Remove CT scans, where the lungs are not fully visible 
# 3. Crop CT scans to the chest region
# <br>
# <br>
# <br>
# <br>
#

# ## 4. Use Body Part Regression for Preprocessing
# https://github.com/MIC-DKFZ/BodyPartRegression
# <img src="images/body-part-regression-explanation.png" width=1000/>
# <br> 
# <br> 
# <br> 
# <br> 
# <br>
# <br>
# <br>

# ## 5. Create body part metadata file for each CT image 
# <img src="images/Main-body-part-regression-function.png" width=800/>
#

# +
# bpreg_inference(nifti_path, json_filepath, plot=True)
# -

# ## 6. Analyze body part meta data files

plot_scores_interactive(json_path, nifti_path)

# +
df = pd.DataFrame()
for i, file in tqdm(enumerate([f for f in os.listdir(json_path) if f.endswith(".json")])): 
    with open(json_path + file, "rb") as f:  
        myDict = json.load(f)
        df.loc[i, "FILE"] = file
        df.loc[i, "BODY PART"] = myDict["body part examined tag"]

df_shapes = pd.read_csv("shapes.csv", index_col=0)
df = df.merge(df_shapes)
# Filter CT scans with less than 20 slices 
df = df[df.z > 20]

# -

bodyparts = plot_dicomexamined_distribution(df, column="BODY PART", count_column="FILE", others_percentage_upper_bound=0.015)

# ## 7. Remove invalid CT scans
# - Filter CT scans, were the predicted body part is **NONE** 

print(f"Corrupted CT scans: {len(df[df['BODY PART'] == 'NONE'])}") 
df2 = df[df['BODY PART'] != 'NONE']
print(f"Dataset size after removing corrupted CT scans: {len(df2)}") 


bodyparts = plot_dicomexamined_distribution(df2, column="BODY PART", count_column="FILE", others_percentage_upper_bound=0.02)

# ## 8. Filter chest CT scans

print(f"CT scans where the chest was not imaged:  {len(df2[~df2['BODY PART'].str.contains('CHEST')])}") 
df2 = df2[df2['BODY PART'].str.contains('CHEST')]
print(f"Dataset after removing the CHEST body part:  {len(df2)}") 

bodyparts = plot_dicomexamined_distribution(df2, column="BODY PART", count_column="FILE", others_percentage_upper_bound=0.02)

# ## 9. Crop chest region out of CT scan

# <img src="images/landmarks-anatomy-2.png" width=600/>
# Adapted from: 
# https://www.freepik.com/free-vector/set-human-body-anatomy_10163663.htm

# +
json_filepaths = [json_path + f for f in os.listdir(json_path) if f.endswith(".json")]
x = load_json(json_filepaths[0])
lookuptable = pd.DataFrame(x["look-up table"]).T
start_score = x["look-up table"]["lung_start"]["mean"]
end_score = x["look-up table"]["lung_end"]["mean"]

lookuptable.sort_values(by="mean")[["mean"]]

# +
nifti_filepaths = [nifti_path + f.replace(".json", ".nii.gz") for f in df2.FILE]

# crop and save ct images to chest region 
# crop_ct_images(nifti_filepaths, json_path,  cropped_path, save=True)

# create body part meta data for cropped images
# bpreg_inference(cropped_path, cropped_json_path, plot=True, gpu_available=False)
# -

# ## 10. Analyze preprocessed dataset 

# +
df = pd.DataFrame()
for i, file in tqdm(enumerate([f for f in os.listdir(cropped_json_path) if f.endswith(".json")])): 
    with open(cropped_json_path + file, "rb") as f:  
        myDict = json.load(f)
        df.loc[i, "FILE"] = file
        df.loc[i, "BODY PART"] = myDict["body part examined tag"]

df_shapes = pd.read_csv("shapes.csv", index_col=0)
df = df.merge(df_shapes)

# -

bodyparts = plot_dicomexamined_distribution(df, column="BODY PART", count_column="FILE", others_percentage_upper_bound=0.02)

plot_scores_interactive(cropped_json_path, cropped_path)









# ------------------------
# ### You have similar problems in your project?
# ### Feel free to reach out. 
#


