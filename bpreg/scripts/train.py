"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random, pickle, datetime, os, sys, cv2, json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import albumentations as A
from scipy.stats import pearsonr
from tqdm import tqdm


import torch, torchvision
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


sys.path.append("../")
from bpreg.network_architecture.bpr_model import BodyPartRegression
from bpreg.network_architecture.ssbr_model import SSBR
from bpreg.utils.training_utils import *
from bpreg.settings import ModelSettings

cv2.setNumThreads(1)
np.seterr(divide="ignore", invalid="ignore")


def train_config(config: ModelSettings):
    seed_everything(config.random_seed)

    # load data
    df = get_dataframe(config)

    if config.model == "SSBR":
        train_dataloader, val_dataloader, _ = data_preprocessing_ssbr(df, config)
        model = SSBR(alpha=config.alpha, lr=config.lr)

    else:
        train_dataset, val_dataset, _ = get_datasets(config, df)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=22,
            shuffle=config.shuffle_train_dataloader,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, num_workers=22
        )

        # run model
        run_fast_dev(config, train_dataloader, val_dataloader)

        model = BodyPartRegression(
            alpha=config.alpha,
            lr=config.lr,
            lambda_=config.lambda_,
            alpha_h=config.alpha_h,
            beta_h=config.beta_h,
            base_model=config.base_model,
            loss_order=config.loss_order,
        )

    logger_uar = TensorBoardLogger(save_dir=config.save_dir, name=config.model_name)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.epochs,
        precision=16,
        logger=logger_uar,
        deterministic=config.deterministic,
        accumulate_grad_batches=int(config.effective_batch_size / config.batch_size),
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    if config.save_model:
        save_model(model, config, path=logger_uar.log_dir + "/")


def train_json(json_path: str):
    config = ModelSettings()
    config.load(json_path)
    print(config)
    train_config(config)


def train_json_list(config_filepaths: list):
    """Train for each config in config_filepaths a model.

    Args:
        config_filepaths (list): list of paths to config-files
    """

    # run code for different configurations
    for filepath in config_filepaths:
        train_json(filepath)


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", nargs="+", default=[])
    parser.add_argument(
        "--path",
        default="/home/AD/s429r/Documents/Code/bodypartregression/src/configs/local/test/",
    )

    value = parser.parse_args()
    config_filenames = value.list
    config_filepath = value.path

    # if no filenames are defined use all files from path
    if len(config_filenames) == 0:
        config_filenames = [
            f for f in np.sort(os.listdir(config_filepath)) if f.endswith(".json")
        ]

    config_filepaths = [config_filepath + file for file in config_filenames]
    train_json_list(config_filepaths)
