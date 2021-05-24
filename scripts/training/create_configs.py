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
    "model_name": "order-loss-experiment", # TODO
    "save_dir": save_path[mode], 
    "shuffle_train_dataloader": True,
    "random_seed": 0, 
    "deterministic": True, 
    "save_model": True, # TODO 
    "base_model": "vgg", 
    "   ": "\n*******************************************************", 

    "batch_size": 21, # TODO
    "effective_batch_size": 21, # TODO 
    "equidistance_range": [5, 100], # in mmm 
    "num_slices": 12, 

    "    ": "\n*******************************************************", 
    "alpha_h": 1, 
    "beta_h": 0.01,
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
    
    
    experiments = {0: {'alpha_h': 1, 'beta_h': 0.001, 'name': 'loh-1a-0.001b.p', 'model_name':'loh-experiment'},
                   1: {'alpha_h': 1, 'beta_h': 0.01, 'name': 'loh-1a-0.01b.p', 'model_name':'loh-experiment'},
                   2: {'alpha_h': 1, 'beta_h': 0.1, 'name': 'loh-1a-0.1b.p', 'model_name':'loh-experiment'},
                   3: {"alpha_h": 1, "beta_h": 0.0001, "name": "loh-1a-0.0001b.p", 'model_name':'loh-experiment'},
                   4: {"alpha_h": 1, "beta_h": 1, "name": "loh-1a-1b.p", 'model_name':'loh-experiment'}}
    
    experiments = {0: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'h',
                      'alpha': 0,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-8m-0.000a.p'},
                     1: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'h',
                      'alpha': 0.005,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-8m-0.005a.p'},
                     2: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'h',
                      'alpha': 0.01,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-8m-0.010a.p'},
                     3: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'h',
                      'alpha': 0.05,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-8m-0.050a.p'},
                     4: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'h',
                      'alpha': 0,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-12m-0.000a.p'},
                     5: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'h',
                      'alpha': 0.005,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-12m-0.005a.p'},
                     6: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'h',
                      'alpha': 0.01,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-12m-0.010a.p'},
                     7: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'h',
                      'alpha': 0.05,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-12m-0.050a.p'},
                     8: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'h',
                      'alpha': 0,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-16m-0.000a.p'},
                     9: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'h',
                      'alpha': 0.005,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-16m-0.005a.p'},
                     10: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'h',
                      'alpha': 0.01,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-16m-0.010a.p'},
                     11: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'h',
                      'alpha': 0.05,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-h-16m-0.050a.p'},
                     12: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'c',
                      'alpha': 0,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-8m-0.0a.p'},
                     13: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'c',
                      'alpha': 0.8,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-8m-0.8a.p'},
                     14: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'c',
                      'alpha': 1,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-8m-1.0a.p'},
                     15: {'num_slices': 8,
                      'batch_size': 32,
                      'effective_batch_size': 32,
                      'epochs': 240,
                      'loss_order': 'c',
                      'alpha': 1.2,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-8m-1.2a.p'},
                     16: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'c',
                      'alpha': 0,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-12m-0.0a.p'},
                     17: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'c',
                      'alpha': 0.8,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-12m-0.8a.p'},
                     18: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'c',
                      'alpha': 1,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-12m-1.0a.p'},
                     19: {'num_slices': 12,
                      'batch_size': 21,
                      'effective_batch_size': 21,
                      'epochs': 160,
                      'loss_order': 'c',
                      'alpha': 1.2,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-12m-1.2a.p'},
                     20: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'c',
                      'alpha': 0,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-16m-0.0a.p'},
                     21: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'c',
                      'alpha': 0.8,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-16m-0.8a.p'},
                     22: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'c',
                      'alpha': 1,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-16m-1.0a.p'},
                     23: {'num_slices': 16,
                      'batch_size': 16,
                      'effective_batch_size': 16,
                      'epochs': 120,
                      'loss_order': 'c',
                      'alpha': 1.2,
                      'model_name': 'order-loss-experiment',
                      'name': 'order-loss-c-16m-1.2a.p'}}


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




