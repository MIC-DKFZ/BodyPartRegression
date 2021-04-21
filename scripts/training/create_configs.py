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

# Array-3.5 mm pixel spacing setup
custom_transform = transforms.Compose([GaussNoise(var_min=0, var_max=20, p=0.5),
                                        ShiftHU(limit=10, p=0.5, min_value=0, max_value=255), # shift HU äquivalent to +- 100 HU
                                        AddFrame(p=0.25), 
                                        ScaleHU(scale_delta=0.2, p=0.5)]) 

albumentation_transform =  A.Compose([A.Flip(p=0.5), # Flip the input either horizontally, vertically or both
                                       A.Transpose(p=0.5),
                                       A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, 
                                                          rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                                       A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0.5, always_apply=False, p=0.5)])

                                      

    
custom_transform_params = {}
for i in range(len(custom_transform.__dict__["transforms"])): 
        custom_transform_params[custom_transform.__dict__["transforms"][i].__class__] =  custom_transform.__dict__["transforms"][i].__dict__

albumentation_transforms_params = albumentation_transform.__dict__["transforms"].transforms



albumentation_transforms_params 


data_path = {"local": "/home/AD/s429r/Documents/Data/DataSet/", 
             "cluster": "/gpu/data/OE0441/s429r/"}

save_path = {"local":"/home/AD/s429r/Documents/Data/Results/body-part-regression-models/", 
             "cluster": "/gpu/checkpoints/OE0441/s429r/results/bodypartregression/"}


mode = "local"

config = {
    "custom_transform": custom_transform, 
    "custom_transform_params": custom_transform_params, 
    "albumentation_transform": albumentation_transform,
    "albumentation_transform_params": albumentation_transforms_params, 
    
    " ": "\n*******************************************************", 
    "df_data_source_path": data_path[mode] + "MetaData/meta-data-public-dataset-npy-arrays-3.5mm-windowing-sigma.xlsx", 
    "data_path":  data_path[mode] + "Arrays-3.5mm-sigma/",
    "landmark_path": data_path[mode] + "MetaData/landmark-meta-data-public.xlsx",
    "model_name": "bodypartregression", # TODO
    "save_dir": save_path[mode], 
    "shuffle_train_dataloader": True,
    "random_seed": 0, 
    "deterministic": True, 
    "save_model": True, # TODO 
    "base_model": "vgg", 
    "   ": "\n*******************************************************", 

    "batch_size": 21,
    "effective_batch_size": 21, 
    "equidistance_range": [5, 200], # in mmm 
    "num_slices": 12, 

    "    ": "\n*******************************************************", 
    "alpha_h": 0.4, 
    "beta_h": 0.02,
    "loss_order": "h", 
    "lambda": 0.0001, 
    "alpha": 0.2,    
    "lr": 1e-4, 
    "epochs": 200, # TODO 
    
    "     ": "\n*******************************************************", 
    "description": "", # TODO
    "pre-name": "standard-config" # TODO 
}
# -

config["accumulate_grad_batches"] = int(config["effective_batch_size"]/config["batch_size"])
#z_range_min = (config["num_slices"]-1) * min(config["equidistance_range"])/10 + 4
#z_range_max = (config["num_slices"]-1) * max(config["equidistance_range"])/10 
#config["z_range_min"] = z_range_min
#config["z_range_max"] = z_range_max
data_path = config["data_path"]


if __name__ == "__main__": 
    datestr = datetime.datetime.now().strftime("%y%m%d-%H:%M")
    config["name"] = config["pre-name"] + ".p"
    print(config["name"])
    with open("../../src/configs/" + mode + "/" + config["name"], 'wb') as f:
        pickle.dump(config, f)




