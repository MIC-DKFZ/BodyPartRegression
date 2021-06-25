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

data_path = {"local": "/home/AD/s429r/Documents/Data/DataSet/", 
            "cluster": "/gpu/data/OE0441/s429r/"}

save_path = {"local":"/home/AD/s429r/Documents/Data/Results/body-part-regression-models/", 
            "cluster": "/gpu/checkpoints/OE0441/s429r/results/bodypartregression/"}

def get_basic_config(mode="cluster", size=128): 
    # Array-3.5 mm pixel spacing setup - 01 scaling
    gaussian_noise = GaussNoise(std_min=0, std_max=0.04, min_value=-1, max_value=1, p=0.5) # equivalent to max gaussian std noise of 50 HU 
    shift_hu = ShiftHU(shift_limit=0.08, min_value=-1, max_value=1, p=0.5) # equivalent to max shift of 100 HU 
    scale_hu = ScaleHU(scale_delta=0.2, min_value=-1, max_value=1, p=0.5)
    add_frame = AddFrame(p=0.25, r_circle=0.75, fill_value=-1, dimension=size)
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





    config = {
        "custom_transform": custom_transform, 
        "custom_transform_params": custom_transform_params, 
        "albumentation_transform": albumentation_transform,
        "albumentation_transform_params": albumentation_transforms_params, 
        
        " ": "\n*******************************************************", 
        "df_data_source_path": data_path[mode] + "MetaData/meta-data-public-dataset-npy-arrays-3.5mm-windowing-sigma.xlsx", 
        "data_path":  data_path[mode] + "Arrays-3.5mm-sigma-01/",
        "landmark_path": data_path[mode] + "MetaData/landmarks-meta-data-v2.xlsx",
        "model_name": "loh-experiment",
        "save_dir": save_path[mode], 
        "shuffle_train_dataloader": True,
        "random_seed": 0, 
        "deterministic": True, 
        "save_model": True, 
        "base_model": "vgg", 
        "   ": "\n*******************************************************", 

        "batch_size": 64, 
        "effective_batch_size": 64, 
        "equidistance_range": [5, 100], 
        "num_slices": 4, 

        "    ": "\n*******************************************************", 
        "alpha_h": 1, 
        "beta_h": 0.01,
        "loss_order": "h", 
        "lambda": 0, 
        "alpha": 0,   
        "lr": 1e-4, 
        "epochs": 480, 
        
        "     ": "\n*******************************************************", 
        "description": "",
        "name": "standard-config.p" 
    }

    config["accumulate_grad_batches"] = int(config["effective_batch_size"]/config["batch_size"])

    return config 


if __name__ == "__main__":

    """

    experiments = {0: {'num_slices': 4,
                        'batch_size': 64,
                        'effective_batch_size': 64,
                        'epochs': 480,
                        'alpha': 1,
                        'lambda': 0,
                        'loss_order': '',
                        'model_name': 'order-loss-experiment',
                        'name': 'no-order-loss-4m.p'},
                        1: {'num_slices': 8,
                        'batch_size': 32,
                        'effective_batch_size': 32,
                        'epochs': 240,
                        'alpha': 1,
                        'lambda': 0,
                        'loss_order': '',
                        'model_name': 'order-loss-experiment',
                        'name': 'no-order-loss-8m.p'},
                        2: {'num_slices': 12,
                        'batch_size': 21,
                        'effective_batch_size': 21,
                        'epochs': 160,
                        'alpha': 1,
                        'lambda': 0,
                        'loss_order': '',
                        'model_name': 'order-loss-experiment',
                        'name': 'no-order-loss-12m.p'}}

    transfomations = {0: {"custom_transform_list":[gaussian_noise, shift_hu, scale_hu, add_frame], 
                        "albumentation_transform_list":[gaussian_blure], 
                        "name": "no-phyiscal-transform.p"}, 
                    1: {"custom_transform_list":[shift_hu, scale_hu, add_frame], 
                        "albumentation_transform_list": [flip, transpose, shift_scale_rotate], 
                        "name": "no-quality-change.p"},
                    2: {"custom_transform_list": [gaussian_noise, shift_hu, scale_hu],
                        "albumentation_transform_list": [flip, transpose, shift_scale_rotate, gaussian_blure], 
                        "name": "no-add-frame.p"}, 
                    3: {"custom_transform_list": [gaussian_noise, add_frame], 
                        "albumentation_transform_list": [flip, transpose, shift_scale_rotate, gaussian_blure], 
                        "name": "no-hu-transform.p"}, 
                    4: {"custom_transform_list": [], 
                        "albumentation_transform_list": [], 
                        "name": "no-transform.p"}}
    experiments = {i: {} for i in range(5)}
    for i, myDict in transfomations.items(): 
        custom_transform = transforms.Compose(myDict["custom_transform_list"])
        albumentation_transform = A.Compose(myDict["albumentation_transform_list"])
        
        custom_transform_params = {}
        for j in range(len(custom_transform.__dict__["transforms"])): 
            custom_transform_params[custom_transform.__dict__["transforms"][j].__class__] =  custom_transform.__dict__["transforms"][j].__dict__

        albumentation_transforms_params = albumentation_transform.__dict__["transforms"].transforms
        
        experiments[i]["custom_transform"] = custom_transform
        experiments[i]["albumentation_transform"] = albumentation_transform
        experiments[i]["custom_transform_params"] = custom_transform_params
        experiments[i]["albumentation_transform_params"] = albumentation_transforms_params   
        experiments[i]["name"] = myDict["name"]                 

    # SSBR Experiment
    # Data Augmentation 
    config["albumentation_transform"] = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.3, scale_limit=(0, 1.5), rotate_limit=0, p=1),
    ])
    config["albumentation_transform_params"] = {"shift_limit": 0.3, "scale_limit": (0, 1.5), "p": 1, "rotate_limit": 0}
    config["custom_transform"] =  transforms.Compose([])
    config["custom_transform_params"] = {}
    config["model"] = "SSBR" # TODO 


    experiments = {0: {'num_slices': 4,
                        'batch_size': 64,
                        'effective_batch_size': 64,
                        'epochs': 480,
                        'alpha': 0.8,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-4m-0.8a.p'},
                        1: {'num_slices': 4,
                        'batch_size': 64,
                        'effective_batch_size': 64,
                        'epochs': 480,
                        'alpha': 1,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-4m-1a.p'},
                        2: {'num_slices': 4,
                        'batch_size': 64,
                        'effective_batch_size': 64,
                        'epochs': 480,
                        'alpha': 1.2,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-4m-1.2a.p'},
                        3: {'num_slices': 8,
                        'batch_size': 32,
                        'effective_batch_size': 32,
                        'epochs': 240,
                        'alpha': 0.8,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-8m-0.8a.p'},
                        4: {'num_slices': 8,
                        'batch_size': 32,
                        'effective_batch_size': 32,
                        'epochs': 240,
                        'alpha': 1,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-8m-1a.p'},
                        5: {'num_slices': 8,
                        'batch_size': 32,
                        'effective_batch_size': 32,
                        'epochs': 240,
                        'alpha': 1.2,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-8m-1.2a.p'},
                        6: {'num_slices': 12,
                        'batch_size': 21,
                        'effective_batch_size': 21,
                        'epochs': 160,
                        'alpha': 0.8,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-12m-0.8a.p'},
                        7: {'num_slices': 12,
                        'batch_size': 21,
                        'effective_batch_size': 21,
                        'epochs': 160,
                        'alpha': 1,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-12m-1a.p'},
                        8: {'num_slices': 12,
                        'batch_size': 21,
                        'effective_batch_size': 21,
                        'epochs': 160,
                        'alpha': 1.2,
                        'model_name': 'ssbr-experiment',
                        'name': 'ssbr-12m-1.2a.p'}}




    experiments = {0: {'alpha_h': 1, 'beta_h': 0.001, 'name': 'loh-1a-0.001b-m4.p', 'model_name':'loh-experiment'},
                   1: {'alpha_h': 1, 'beta_h': 0.01, 'name': 'loh-1a-0.01b-m4.p', 'model_name':'loh-experiment'},
                   2: {'alpha_h': 1, 'beta_h': 0.1, 'name': 'loh-1a-0.1b-m4.p', 'model_name':'loh-experiment'},
                   3: {"alpha_h": 1, "beta_h": 0.0001, "name": "loh-1a-0.0001b-m4.p", 'model_name':'loh-experiment'},
                   4: {"alpha_h": 1, "beta_h": 1, "name": "loh-1a-1b-m4.p", 'model_name':'loh-experiment'},
                   5: {'alpha_h': 1, 'beta_h': 0.02, 'name': 'loh-1a-0.02b-m4.p', 'model_name':'loh-experiment'},
                   6: {'alpha_h': 1, 'beta_h': 0.03, 'name': 'loh-1a-0.03b-m4.p', 'model_name':'loh-experiment'},
                   7: {'alpha_h': 1, 'beta_h': 0.04, 'name': 'loh-1a-0.04b-m4.p', 'model_name':'loh-experiment'},
                   8: {'alpha_h': 1, 'beta_h': 0.009, 'name': 'loh-1a-0.009b-m4.p', 'model_name':'loh-experiment'},
                   9: {'alpha_h': 1, 'beta_h': 0.008, 'name': 'loh-1a-0.008b-m4.p', 'model_name':'loh-experiment'},
                   10: {'alpha_h': 1, 'beta_h': 0.007, 'name': 'loh-1a-0.007b-m4.p', 'model_name':'loh-experiment'}}
    """ 
    mode = "cluster"
    experiments = {0: {"model_name": "public-model",
                       "name": "public-model-v1.p", 
                       "df_data_source_path": data_path[mode] + "MetaData/meta-data-publish-model.xlsx", 
                       "data_path":  data_path[mode] + "Arrays-3.5mm-sigma-01/",
                       "landmark_path": data_path[mode] + "MetaData/landmarks-publish-model.xlsx"}}
    for idx, data in experiments.items(): 
        config = get_basic_config(mode=mode)
        print("Idx: ", idx)
        for key in data.keys(): 
            config[key] = data[key]
            print(key, data[key])

        save_path_folder = "../../src/configs/" + mode + "/" +  config["model_name"] + "/" 
        if not os.path.exists(save_path_folder): os.mkdir(save_path_folder)
        save_path = save_path_folder + config["name"]
        
        # Save file 
        with open(save_path, 'wb') as f:
            print("save file: ", save_path)
            pickle.dump(config, f)




