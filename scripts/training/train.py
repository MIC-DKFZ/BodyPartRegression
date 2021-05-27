import random, pickle, datetime, os, sys, cv2
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import argparse
import albumentations as A
from scipy.stats import pearsonr
from tqdm import tqdm

cv2.setNumThreads(1)

import torch, torchvision
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint



sys.path.append("../../")
from scripts.network_architecture.bpr_model import BodyPartRegression
from scripts.dataset.bpr_dataset import BPRDataset
from scripts.postprocessing.lookup import LookUp

np.seterr(divide='ignore', invalid='ignore') #TODO

def get_dataframe(config): 
    df = pd.read_excel(config["df_data_source_path"], sheet_name="data", engine='openpyxl')
    df = df[(df["z"] >= 30)]
    return df


def get_datasets(config, df):
    train_filenames = df.loc[df.train_data == 1, "filename"].values
    val_filenames =   df.loc[df.val_data == 1, "filename"].values
    test_filenames =  df.loc[df.test_data == 1, "filename"].values

    train_zspacings = df.loc[df.train_data == 1, "pixel_spacingz"].values
    val_zspacings =   df.loc[df.val_data == 1, "pixel_spacingz"].values
    test_zspacings =  df.loc[df.test_data == 1, "pixel_spacingz"].values
    
    
    train_dataset = BPRDataset(data_path=config["data_path"], 
                               filenames=train_filenames, 
                                z_spacings=train_zspacings, 
                                landmark_path=config["landmark_path"], 
                                landmark_sheet_name="landmarks-train", # TODO -without-merge
                                random_seed=config["random_seed"], 
                                custom_transform=config["custom_transform"], 
                                albumentation_transform=config["albumentation_transform"],
                                equidistance_range=config["equidistance_range"], 
                                num_slices=config["num_slices"])
    
    
    val_dataset = BPRDataset(data_path=config["data_path"], 
                             filenames=val_filenames, 
                             z_spacings=val_zspacings,
                            landmark_path=config["landmark_path"], 
                            landmark_sheet_name="landmarks-val",                         
                             random_seed=config["random_seed"],  
                             custom_transform=config["custom_transform"], 
                             albumentation_transform=config["albumentation_transform"],
                             equidistance_range=config["equidistance_range"], 
                             num_slices=config["num_slices"])
    
    test_dataset = BPRDataset(data_path=config["data_path"], 
                              filenames=test_filenames, 
                                z_spacings=test_zspacings,
                                landmark_path=config["landmark_path"], 
                                landmark_sheet_name="landmarks-test",                          
                                random_seed=config["random_seed"],
                                custom_transform=config["custom_transform"], 
                                albumentation_transform=config["albumentation_transform"],
                                equidistance_range=config["equidistance_range"], 
                                num_slices=config["num_slices"])
    
    return train_dataset, val_dataset, test_dataset


def run_fast_dev(config, train_dataloader, val_dataloader): 
    model = BodyPartRegression(alpha=config["alpha"], 
             lr=config["lr"], 
             lambda_=config["lambda"], 
             alpha_h=config["alpha_h"], 
             beta_h=config["beta_h"],
             base_model=config["base_model"], 
             loss_order=config["loss_order"])
    trainer_dev =  pl.Trainer(gpus=1, fast_dev_run=True, precision=16)
    trainer_dev.fit(model, train_dataloader, val_dataloader)

def save_model(trainer, model, config, train_dataloader, logger): 
    # save lookuptable
    lookup = LookUp(model, train_dataloader.dataset)

    log_dir = logger.log_dir + "/"
    print("save model at: ", log_dir)

    lookup.save(log_dir)
    with open(log_dir + 'config.p', 'wb') as f:
        pickle.dump(config, f)

    if config["save_model"]: torch.save(model.state_dict(), log_dir + "model.pt") 


def data_preprocessing_ssbr(df, config): 
    train_filenames = df[df.train_data == 1].filename.values
    val_filenames = df[df.val_data == 1].filename.values
    test_filenames = df[df.test_data == 1].filename.values

    train_filepaths = [config["data_path"] + f for f in train_filenames]
    val_filepaths = [config["data_path"] + f for f in val_filenames]
    test_filepaths = [config["data_path"] + f for f in test_filenames]


    train_zspacings = df[df.filename.isin(train_filenames)]["pixel_spacingz"].values
    val_zspacings = df[df.filename.isin(val_filenames)]["pixel_spacingz"].values
    test_zspacings = df[df.filename.isin(test_filenames)]["pixel_spacingz"].values

    train_dataset = SSBRDataset(train_filepaths,  
                        landmark_path=config["landmark_path"], 
                        landmark_sheet_name="landmarks-train", # TODO -without-merge
                        random_seed=config["random_seed"], 
                        custom_transform=config["custom_transform"], 
                        albumentation_transform=config["albumentation_transform"],
                        equidistance_range=config["equidistance_range"], 
                        num_slices=config["num_slices"])


    val_dataset = SSBRDataset(val_filepaths,  
                            landmark_path=config["landmark_path"], 
                            landmark_sheet_name="landmarks-val",                         
                            random_seed=config["random_seed"],  
                            custom_transform=config["custom_transform"], 
                            albumentation_transform=config["albumentation_transform"],
                            equidistance_range=config["equidistance_range"], 
                            num_slices=config["num_slices"])

    test_dataset = SSBRDataset(test_filepaths, 
                        landmark_path=config["landmark_path"], 
                        landmark_sheet_name="landmarks-test",                          
                        random_seed=config["random_seed"],
                        custom_transform=config["custom_transform"], 
                        albumentation_transform=config["albumentation_transform"],
                        equidistance_range=config["equidistance_range"], 
                        num_slices=config["num_slices"])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=config["batch_size"], 
                                                num_workers=20, 
                                                shuffle=config["shuffle_train_dataloader"])

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=config["batch_size"], 
                                                num_workers=20)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=config["batch_size"], 
                                                num_workers=20)

    return train_dataloader, val_dataloader, test_dataloader

def train_config(config): 
    # print configurations 
    seed_everything(config["random_seed"])

    print("CONFIGURATION")
    print("*******************************************************\n")
    for key, item in config.items(): 
        if key.startswith(("custom_transform", "albumentation_transform")): continue
        print(f"{key:<30}\t{item}")

    # load data 
    df = get_dataframe(config)

    if "model" in config.keys(): 
        if config["model"] != "SSBR": train_dataloader, val_dataloader, test_dataloader = data_preprocessing_ssbr(df, config)
    
    else: 
        train_dataset, val_dataset, test_dataset = get_datasets(config, df)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=config["batch_size"], 
                                                num_workers=22, 
                                                shuffle=config["shuffle_train_dataloader"])

        val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                    batch_size=config["batch_size"], 
                                                    num_workers=22)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                    batch_size=config["batch_size"], 
                                                    num_workers=22)

        # run model 
        run_fast_dev(config, train_dataloader, val_dataloader)
    
    model = BodyPartRegression(alpha=config["alpha"], 
                                lr=config["lr"], 
                                lambda_=config["lambda"], 
                                alpha_h=config["alpha_h"], 
                                beta_h=config["beta_h"],
                                base_model=config["base_model"], 
                                loss_order=config["loss_order"])    


    logger_uar = TensorBoardLogger(save_dir=config["save_dir"], name=config["model_name"])
    checkpoint_callback = ModelCheckpoint(monitor='mse')
    trainer = pl.Trainer(gpus=1, 
                         callbacks=[checkpoint_callback],
                         max_epochs=config["epochs"], 
                         precision=16,
                         logger=logger_uar, deterministic=config["deterministic"], 
                         # val_check_interval=0.25, log_every_n_steps=25,
                         accumulate_grad_batches=int(config["effective_batch_size"]/config["batch_size"])) 

    trainer.fit(model, train_dataloader, val_dataloader) 
    save_model(trainer, model, config, train_dataloader, logger_uar)

def train(): 
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", nargs="+", default=[])
    parser.add_argument('--path', default="/home/AD/s429r/Documents/Code/s429r/trainings/configs/test/" )
    sys.path.append("../../../s429r/") # TODO Pfad ändern! 
    value = parser.parse_args()
    config_filenames = value.list
    config_filepath = value.path 
    
    if len(config_filenames) == 0: 
        config_filenames = np.sort(os.listdir(config_filepath))
    
    # run code for different configurations 
    for filename in config_filenames:
        with open(config_filepath + filename, 'rb') as f:
            config = pickle.load(f)
        
        train_config(config)

if __name__ == "__main__": 
    train() 

