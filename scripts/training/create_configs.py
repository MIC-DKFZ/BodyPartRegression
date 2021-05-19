# -*- coding: utf-8 -*-
import albumentations as A
import numpy as np 
import datetime
import sys
import os
import pickle
import cv2
cv2.setNumThreads(1)

from torchvision import transforms

sys.path.append("../../")
from scripts.dataset.custom_transformations import *

# Array-3.5 mm pixel spacing setup - 01 scaling
gaussian_noise = GaussNoise(std_min=0, std_max=0.04, min_value=-1, max_value=1, p=0.5) # equivalent to max gaussian std noise of 50 HU 
shift_hu = ShiftHU(shift_limit=0.08, min_value=-1, max_value=1, p=0.5) # equivalent to max shift of 100 HU 
scale_hu = ScaleHU(scale_delta=0.2, min_value=-1, max_value=1, p=0.5)
add_frame = AddFrame(p=0.25, r_circle=0.75, fill_value=-1)
flip = A.Flip(p=0.5) # Flip the input either horizontally, vertically or both
transpose = A.Transpose(p=0.5) 
shift_scale_rotate = A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=10, p=0.5, 
                                        border_mode=cv2.BORDER_REFLECT_101)
gaussian_blure = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.5, always_apply=False, p=0.5)

custom_transform = transforms.Compose([gaussian_noise, shift_hu, scale_hu, add_frame])
albumentation_transform =  A.Compose([flip, transpose, shift_scale_rotate, gaussian_blure])


custom_transform_params = {}
for i in range(len(custom_transform.__dict__["transforms"])): 
        custom_transform_params[custom_transform.__dict__["transforms"][i].__class__] =  custom_transform.__dict__["transforms"][i].__dict__

albumentation_transforms_params = albumentation_transform.__dict__["transforms"].transforms


data_path = {"local": "/home/AD/s429r/Documents/Data/DataSet/", 
             "cluster": "/gpu/data/OE0441/s429r/"}

save_path = {"local":"/home/AD/s429r/Documents/Data/Results/body-part-regression-models/", 
             "cluster": "/gpu/checkpoints/OE0441/s429r/results/bodypartregression/"}


mode = "cluster"

config = {
    "custom_transform": custom_transform, 
    "custom_transform_params": custom_transform_params, 
    "albumentation_transform": albumentation_transform,
    "albumentation_transform_params": albumentation_transforms_params, 
    
    " ": "\n*******************************************************", 
    "df_data_source_path": data_path[mode] + "MetaData/meta-data-public-dataset-npy-arrays-3.5mm-windowing-sigma.xlsx", 
    "data_path":  data_path[mode] + "Arrays-3.5mm-sigma-01/",
    "landmark_path": data_path[mode] + "MetaData/landmarks-meta-data-v2.xlsx",
    "model_name": "loh-experiment", # TODO
    "save_dir": save_path[mode], 
    "shuffle_train_dataloader": True,
    "random_seed": 0, 
    "deterministic": True, 
    "save_model": True, # TODO 
    "base_model": "vgg", 
    "   ": "\n*******************************************************", 

    "batch_size": 20, # TODO
    "effective_batch_size": 20, # TODO 
    "equidistance_range": [5, 100], # in mmm 
    "num_slices": 12, 

    "    ": "\n*******************************************************", 
    "alpha_h": 0.4, 
    "beta_h": 0.02,
    "loss_order": "h", 
    "lambda": 0, # 0.0001, 
    "alpha": 0, #0.2,    
    "lr": 1e-4, 
    "epochs": 160, # TODO 
    
    "     ": "\n*******************************************************", 
    "description": "", # TODO
    "name": "standard-config-01.p" # TODO 
}

config["accumulate_grad_batches"] = int(config["effective_batch_size"]/config["batch_size"])


if __name__ == "__main__":
    experiments = {
        0: {"alpha_h": 0.002, "beta_h": 0.0001, "name": "loh-experiment-0.0001b-0.002a.p"},
        1: {"alpha_h": 0.02, "beta_h": 0.001, "name": "loh-experiment-0.001b-0.02a.p"},
        2: {"alpha_h": 0.2, "beta_h": 0.01, "name": "loh-experiment-0.01b-0.2a.p"},
        3: {"alpha_h": 2, "beta_h": 0.1, "name": "loh-experiment-0.1b-2a.p"},
        4: {"alpha_h": 20, "beta_h": 1, "name": "loh-experiment-1b-20a.p"},
    }

    experiments = {
        0: {"alpha_h": 0.1, "beta_h": 0.005, "name": "loh-experiment-0.005b-0.1a.p"},
        1: {"alpha_h": 0.4, "beta_h": 0.02, "name": "loh-experiment-0.02b-0.4a.p"},
        2: {"alpha_h": 0.6, "beta_h": 0.03, "name": "loh-experiment-0.03b-0.6a.p"},
        3: {"alpha_h": 0.8, "beta_h": 0.04, "name": "loh-experiment-0.4b-0.8a.p"},
    }
    
    
    experiments = {0: {'alpha': 0.001, 'beta': 0.001, 'name': 'loh-0.001a-0.001b.p'},
 1: {'alpha': 0.001, 'beta': 0.01, 'name': 'loh-0.001a-0.01b.p'},
 2: {'alpha': 0.001, 'beta': 0.1, 'name': 'loh-0.001a-0.1b.p'},
 3: {'alpha': 0.01, 'beta': 0.001, 'name': 'loh-0.01a-0.001b.p'},
 4: {'alpha': 0.01, 'beta': 0.01, 'name': 'loh-0.01a-0.01b.p'},
 5: {'alpha': 0.01, 'beta': 0.1, 'name': 'loh-0.01a-0.1b.p'},
 6: {'alpha': 0.1, 'beta': 0.001, 'name': 'loh-0.1a-0.001b.p'},
 7: {'alpha': 0.1, 'beta': 0.01, 'name': 'loh-0.1a-0.01b.p'},
 8: {'alpha': 0.1, 'beta': 0.1, 'name': 'loh-0.1a-0.1b.p'},
 9: {'alpha': 1, 'beta': 0.001, 'name': 'loh-1a-0.001b.p'},
 10: {'alpha': 1, 'beta': 0.01, 'name': 'loh-1a-0.01b.p'},
 11: {'alpha': 1, 'beta': 0.1, 'name': 'loh-1a-0.1b.p'},
 12: {'alpha': 0.1, 'beta': 0.0001, 'name': 'loh-0.1a-0.0001b.p'},
 13: {'alpha': 0.1, 'beta': 1, 'name': 'loh-0.1a-1b.p'}}

    save_path_folder = "../../src/configs/" + mode + "/" +  config["model_name"] + "/" 
    if not os.path.exists(save_path_folder): os.mkdir(save_path_folder)

    for idx, data in experiments.items(): 
        print("Idx: ", idx)
        for key in data.keys(): 
            config[key] = data[key]
            print(key, data[key])

        save_path = save_path_folder + config["name"]
        # Save file 
        with open(save_path, 'wb') as f:
            print("save file: ", save_path)
            pickle.dump(config, f)




