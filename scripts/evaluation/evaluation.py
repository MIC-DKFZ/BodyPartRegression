import os, sys, pickle, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from tqdm import tqdm
from scipy import interpolate

import pytorch_lightning as pl
import torch

sys.path.append("../../")
from scripts.network_architecture.bpr_model import BodyPartRegression
from scripts.training.train import get_dataframe, get_datasets
from scripts.evaluation.accuracy import Accuracy
from scripts.evaluation.normalized_mse import NormalizedMSE 
from scripts.evaluation.visualization import Visualization
from scripts.inference.inference_model import InferenceModel
from scripts.score_processing.landmark_scores import LandmarkScoreBundle, LandmarkScores
from src.settings.settings import *

class Evaluation(Visualization):

    def __init__(self, base_filepath,  
                 val_dataset=False, 
                 df_data_source_path="", 
                 landmark_path="", 
                 data_path="", 
                 device="cuda"):
        Visualization.__init__(self)
        self.device = device
        self.base_filepath = base_filepath
        self.inference_model = InferenceModel(base_filepath)
        self.normalizedMSE = NormalizedMSE()
        self.landmark_score_bundle = LandmarkScoreBundle(data_path, landmark_path, self.inference_model.model)

        # setup data
        self.df_data_source_path = df_data_source_path
        self.landmark_path = landmark_path # TODO --> model dataclass with all attributes and _load_model function 
        self.data_path = data_path
        self._setup_data(val_dataset=val_dataset)

        self.mse, self.mse_std = self.landmark_score_bundle.nMSE(target="validation", reference="train")
        self.acc5 = self.landmark_score_bundle.accuracy(self.val_dataset, reference="train", class2landmark=CLASS_TO_LANDMARK_5)
                

    def _setup_data(self, val_dataset=False):
        path = self.base_filepath + "config.p"  # TODO !

        with open(path, "rb") as f:
            config = pickle.load(f)

        config["num_slices"] = 8
        config["batch_size"] = 32
        config["shuffle_train_dataloader"] = False

        self.config = config 

        if len(self.df_data_source_path) > 0: 
            config["df_data_source_path"] = self.df_data_source_path
        if len(self.landmark_path) > 0: 
            config["landmark_path"] = self.landmark_path
        if len(self.data_path) > 0: 
            config["data_path"] = self.data_path

        df_data = get_dataframe(config)
        if val_dataset:
            self.val_dataset = val_dataset
            self.train_dataset, _, self.test_dataset = get_datasets(config, df_data)
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = get_datasets(
                config, df_data
            )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            num_workers=20,
            shuffle=config["shuffle_train_dataloader"],
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=config["batch_size"], num_workers=20
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=config["batch_size"], num_workers=20
        )


    def print_summary(self):
        print("Model summary\n*******************************")

        print(
            f"\nNormalized MSE [1e-3]:\t{self.mse*1e3:<1.3f} +- {self.mse_std*1e3:<1.3f}")
        print(
            f"Accuracy (5 classes) : \t{self.acc5*100:<1.2f}%"
        )
        print("\nLook-up table for training data \n*******************************")
        self.landmark_score_bundle.dict["train"].print_lookuptable()

    def plot_landmarks(self):
        validation_score_matrix = self.landmark_score_bundle.dict["validation"].score_matrix 
        expected_scores = self.landmark_score_bundle.dict["train"].expected_scores

        super(Evaluation, self).plot_landmarks(validation_score_matrix, 
                                               expected_scores=expected_scores) 

if __name__ == "__main__": 
    base_dir = "/home/AD/s429r/Documents/Code/bodypartregression/src/models/loh-ldist-l2/sigma-dataset-v11-v2/"
    modelEval = Evaluation(base_dir)

