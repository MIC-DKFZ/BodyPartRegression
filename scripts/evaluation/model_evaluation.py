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
from scripts.postprocessing.lookup import LookUp
from scripts.training.train import get_dataframe, get_datasets
from scripts.evaluation.basic_evaluation import Evaluation, grid_plot
from scripts.evaluation.accuracy import Accuracy
from scripts.evaluation.normalized_mse import NormalizedMSE 
from src.settings.settings import *

###################### TODO ################################################

class ModelEvaluation:
    """
    Todo:
    - allgemeine Klasse mit Funktionen, die nicht nur auf spezielles Modell angewendet werden kann
    - BPREvaluation soll von dieser Klasse erben
    """

    def __init__(self, base_filepath,  
                 val_dataset=False, 
                 overwrite_df_data_source_path="", 
                 overwrite_landmark_path="", 
                 overwrite_data_path=""):

        self.normalizedMSE = NormalizedMSE()
        self.base_filepath = base_filepath
        self.config_filepath = base_filepath + "config.p"
        self.model_filepath = base_filepath + "model.pt"

        self.overwrite_df_data_source_path = overwrite_df_data_source_path
        self.overwrite_landmark_path = overwrite_landmark_path # TODO --> model dataclass with all attributes and _load_model function 
        self.overwrite_data_path = overwrite_data_path

        # setup model
        with open(self.config_filepath, "rb") as f:
            self.config = pickle.load(f)

        self.model = BodyPartRegression(
            alpha=self.config["alpha"],
            lr=self.config["lr"],
            base_model=self.config["base_model"],
        )

        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()
        self.model.to("cuda")

        # setup data
        self._setup_data(val_dataset=val_dataset)
        self.lookup = LookUp(self.model, self.train_dataset)

        # get train and val slice score matrix
        self.val_score_matrix = self.model.compute_slice_score_matrix(self.val_dataset)
        self.train_score_matrix = self.model.compute_slice_score_matrix(self.train_dataset)

        # get mse 
        self.mse, self.mse_std, self.d = self.normalizedMSE.from_dataset(self.model, self.val_dataset, self.train_dataset)
        
        # calculate accuracy for 5 and 4 distinct classes
        self._set_accuracies()
        

    def _setup_data(self, val_dataset=False):
        path = "/home/AD/s429r/Documents/Code/s429r/trainings/configs/local/standard-config.p"
        path = self.config_filepath  # TODO !

        with open(path, "rb") as f:
            config = pickle.load(f)

        config["num_slices"] = 8
        config["batch_size"] = 32
        config["shuffle_train_dataloader"] = False

        if len(self.overwrite_df_data_source_path) > 0: 
            config["df_data_source_path"] = self.overwrite_df_data_source_path
        if len(self.overwrite_landmark_path) > 0: 
            config["landmark_path"] = self.overwrite_landmark_path
        if len(self.overwrite_data_path) > 0: 
            config["data_path"] = self.overwrite_data_path

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

    def _set_accuracies(self): 
        accuracies_5classes = []
        accuracies_3classes = []
        ids = self.val_dataset.landmark_ids

        expected_scores = np.nanmean(self.train_score_matrix, axis=0) 
        # define accuracy class with 5 classes
        acc5 = Accuracy(expected_scores, CLASS_TO_LANDMARK_5)

        # define accuracy class with 3 classes
        acc3 = Accuracy(expected_scores, CLASS_TO_LANDMARK_3)

        for i in range(0, 100): 
            landmark_positions = self.val_dataset.landmark_matrix[i, :]
            x = self.val_dataset.get_full_volume(ids[i])
            scores = self.model.predict_tensor(torch.tensor(x[:, np.newaxis, :, :]))

            accuracies_5classes.append(acc5.volume(scores, landmark_positions))
            accuracies_3classes.append(acc3.volume(scores, landmark_positions))
            
        self.acc5 = np.nanmean(accuracies_5classes)
        self.acc3 = np.nanmean(accuracies_3classes)

    def print_summary(self):
        print("Model summary\n*******************************")

        print(
            f"\nNormalized MSE [1e-3]:\t{self.mse*1e3:<1.3f} +- {self.mse_std*1e3:<1.3f}")
        print(
            f"\nAccuracy (5 classes): \t{self.acc5*100:<1.2f}%"
        )
        print("\nLookup Table\n*******************************")
        self.lut.print()


if __name__ == "__main__": 
    base_dir = "/home/AD/s429r/Documents/Code/bodypartregression/src/models/loh-ldist-l2/sigma-dataset-v11-v2/"
    modelEval = ModelEvaluation(base_dir)

