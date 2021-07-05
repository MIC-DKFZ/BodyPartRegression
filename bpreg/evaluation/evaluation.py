import os, sys, pickle, random
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append("../../")
from bpreg.training.train import get_dataframe, get_datasets
from bpreg.evaluation.landmark_mse import LMSE
from bpreg.evaluation.visualization import Visualization
from bpreg.inference.inference_model import InferenceModel
from bpreg.score_processing.landmark_scores import LandmarkScoreBundle, LandmarkScores
from bpreg.settings.settings import *


class Evaluation(Visualization):
    def __init__(
        self,
        base_filepath: str,
        val_dataset: None = False,
        df_data_source_path: str = DF_DATA_SOURCE_PATH,
        landmark_path: str = LANDMARK_PATH,
        data_path: str = DATA_PATH,
        device: str = "cuda",
        landmark_start: str = "pelvis_start",
        landmark_end: str = "eyes_end",
    ):
        Visualization.__init__(self)
        self.device = device
        self.base_filepath = base_filepath
        self.inference_model = InferenceModel(base_filepath)
        self.normalizedMSE = LMSE()
        self.landmark_score_bundle = LandmarkScoreBundle(
            data_path,
            landmark_path,
            self.inference_model.model,
            landmark_start=landmark_start,
            landmark_end=landmark_end,
        )

        # setup data
        self.df_data_source_path = df_data_source_path
        self.landmark_path = landmark_path  # TODO --> model dataclass with all attributes and _load_model function
        self.data_path = data_path
        self._setup_data(val_dataset=val_dataset)

        self.mse, self.mse_std = self.landmark_score_bundle.nMSE(
            target="validation", reference="train"
        )
        self.acc5 = self.landmark_score_bundle.accuracy(
            self.val_dataset, reference="train", class2landmark=CLASS_TO_LANDMARK_5
        )

    def _setup_data(self, val_dataset=False):
        path = self.base_filepath + "config.p"  # TODO !

        with open(path, "rb") as f:
            self.config = pickle.load(f)

        config = self.config.copy()
        config["num_slices"] = 8
        config["batch_size"] = 32
        config["shuffle_train_dataloader"] = False

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
            f"\nLandmark Mean Square Error:\t{self.mse:<1.3f} +- {self.mse_std:<1.3f}"
        )
        print(f"Accuracy (5 classes) :        \t{self.acc5*100:<1.2f}%")
        print("\nLook-up table for training data \n*******************************")
        self.landmark_score_bundle.dict["train"].print_lookuptable()

    def plot_landmarks(self, alpha=0.7, target="validation"):
        if target == "validation":
            score_matrix = self.landmark_score_bundle.dict[
                "validation"
            ].score_matrix_transformed
        if target == "test":
            score_matrix = self.landmark_score_bundle.dict[
                "test"
            ].score_matrix_transformed
        expected_scores = self.landmark_score_bundle.dict[
            "train"
        ].expected_scores_transformed

        landmark_names = self.landmark_score_bundle.dict["validation"].landmark_names
        super(Evaluation, self).plot_landmarks(
            score_matrix,
            expected_scores=expected_scores,
            landmark_names=landmark_names,
            alpha=alpha,
        )

    def plot_slices2scores(
        self, max_cols=4, nearby_values=[0, 25, 50, 75, 100], save_path=""
    ):
        _, ax = plt.subplots(len(nearby_values), max_cols, figsize=(14, 16))

        for row, nearby_value in enumerate(nearby_values):
            col = 0
            slice_indices = []
            while col < max_cols:
                idx = np.random.randint(0, len(self.test_dataset))
                if idx in slice_indices:
                    continue
                slice_indices.append(idx)

                filepaths = self.test_dataset.filepaths
                zspacing = self.test_dataset.z_spacings[idx]
                X = np.load(filepaths[idx]).transpose(2, 0, 1)
                scores = self.inference_model.predict_npy_array(X)
                myScores = self.inference_model.parse_scores(scores, zspacing)
                slice_index = np.argmin(
                    np.abs(myScores.original_transformed_values - nearby_value)
                )
                slice_score = myScores.original_transformed_values[slice_index]
                if np.abs(slice_score - nearby_value) > 1:
                    continue

                ax[row, col].imshow(X[slice_index, :, :], cmap="gray")
                ax[row, col].set_title(np.round(slice_score, 2), fontsize=14)
                ax[row, col].set_yticklabels([])
                ax[row, col].set_xticklabels([])
                ax[row, col].set_yticks([])
                ax[row, col].set_xticks([])
                col += 1
        plt.tight_layout()
        if len(save_path) > 0:
            plt.savefig(save_path + "model-evaluation-nearby-slices.png")


if __name__ == "__main__":
    base_dir = "/home/AD/s429r/Documents/Code/bodypartregression/src/models/loh-ldist-l2/sigma-dataset-v11-v2/"
    modelEval = Evaluation(base_dir)