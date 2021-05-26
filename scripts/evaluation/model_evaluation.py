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
        
        """
        # get look-up table
        self.train_lm_summary = self.lut.get_lookup_table(self.train_dataset)

        # get metrics
        self.val_metrics = self.trainer.test(self.model, self.val_dataloader)
        self.train_metrics = self.trainer.test(self.model, self.train_dataloader)

        # get validation_predictions
        self.val_preds, self.val_zs = self.predict_dataset(self.val_dataset, self.model)
        (
            self.val_landmark_preds,
            self.val_landmark_preds_ids,
        ) = self.get_landmark_prediction(self.val_dataset, self.val_preds)

        self.val_acc, self.val_std = self.accuracy(
            self.val_dataset, self.val_preds, self.train_lm_summary
        )
        self.mse, self.mse_std = self.normalized_mse(
            self.val_landmark_preds, self.train_lm_summary
        )

        self.min_value = min(
            [self.train_lm_summary[key]["mean"] for key in self.train_lm_summary.keys()]
        )
        self.max_value = max(
            [self.train_lm_summary[key]["mean"] for key in self.train_lm_summary.keys()]
        )
        # setup trainer
        self.trainer = pl.Trainer(
            gpus=1,
            max_epochs=self.config["epochs"],
            precision=16,
            deterministic=self.config["deterministic"],
        )

        """

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




    def landmarks2score(self, i, dataset, train_results):
        myDict = dataset.landmarks[i]
        index = myDict["dataset_index"]
        slice_idx = myDict["slice_indices"]
        landmarks_idx = myDict["defined_landmarks_i"]

        volume = dataset.get_full_volume(index)
        x = np.arange(min(slice_idx), max(slice_idx) + 1)

        predicted_scores = self.predict_image(volume, x)
        expected_scores = [
            train_results[key]["mean"]
            for key in train_results.keys()
            if key in landmarks_idx
        ]
        errors = [
            train_results[key]["std"]
            for key in train_results.keys()
            if key in landmarks_idx
        ]

        return slice_idx, landmarks_idx, x, predicted_scores, expected_scores, errors

    def print_summary(self):
        print("Model summary\n*******************************")
        print(
            f"Landmark metric for validation set:\t{self.val_metrics[0]['test_landmark_metric_mean']:<1.4f}"
        )
        print(
            f"Landmark metric for train set:     \t{self.train_metrics[0]['test_landmark_metric_mean']:<1.4f}"
        )
        print(
            f"\nValidation accuracy:             \t{self.val_acc*100:<1.2f}% +- {self.val_std*100:<1.2f}%"
        )
        print(
            f"Mean relative deviation (in 1e-3): \t{self.mse*1e3:1.3f} +- {self.mse_std*1e3:1.3f}"
        )
        print("\nTraining-set prediction summary\n*******************************")
        self.lut.print(self.train_lm_summary)

    def mse_for_volume(self, vol_idx):
        """
        Notice: mse values are not normalized
        """
        (
            slice_idx,
            landmarks_idx,
            x,
            y_estimated,
            expected_scores,
            errors,
        ) = self.landmarks2score(vol_idx, self.val_dataset, self.train_lm_summary)
        expected_f = interpolate.interp1d(slice_idx, expected_scores, kind="linear")
        y_expected = expected_f(x)
        mse = np.mean(np.sqrt((np.array(y_estimated) - np.array(y_expected)) ** 2))

        return mse

if __name__ == "__main__": 
    base_dir = "/home/AD/s429r/Documents/Code/bodypartregression/src/models/loh-ldist-l2/sigma-dataset-v11-v2/"
    modelEval = ModelEvaluation(base_dir)

