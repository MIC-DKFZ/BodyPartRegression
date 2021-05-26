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
from scripts.postprocessing.lookuptable import LookUpTable
from scripts.training.train import get_dataframe, get_datasets
from scripts.evaluation.basic_evaluation import Evaluation, grid_plot
from src.settings.settings import *

###################### TODO ################################################

class ModelEvaluation(Evaluation):
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

        Evaluation.__init__(self)
        self.lut = LookUpTable(base_filepath)
        self.base_filepath = base_filepath
        self.config_filepath = base_filepath + "config.p"
        self.model_filepath = base_filepath + "model.pt"

        self.overwrite_df_data_source_path = overwrite_df_data_source_path
        self.overwrite_landmark_path = overwrite_landmark_path
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

        # setup trainer
        self.trainer = pl.Trainer(
            gpus=1,
            max_epochs=self.config["epochs"],
            precision=16,
            deterministic=self.config["deterministic"],
        )

        # setup data
        self._setup_data(val_dataset=val_dataset)

        # get train and val slice score matrix
        self.val_score_matrix = self.model.compute_slice_score_matrix(self.val_dataset)
        self.train_score_matrix = self.model.compute_slice_score_matrix(self.train_dataset)
        self.mse, self.mse_std, self.d = self.model.normalized_mse(self.val_dataset, self.train_dataset)
        
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

    def get_landmark_xy(self, dataset, preds, landmark):
        landmark_xy = {}
        for i, myDict in dataset.landmarks.items():
            index = myDict["dataset_index"]
            if not index in preds.keys():
                continue

            ys = preds[index]
            if not landmark in myDict["defined_landmarks_i"]:
                continue
            x = myDict["slice_indices"][
                np.where(myDict["defined_landmarks_i"] == landmark)[0][0]
            ]
            y = ys[x]

            landmark_xy[index] = {}
            landmark_xy[index]["x"] = x
            landmark_xy[index]["y"] = y
        return landmark_xy

    def plot_score2index_xyfix(
        self,
        dataset,
        preds,
        zs,
        landmark_anchor=3,
        ids=np.arange(0, 20),
        save_path=False,
    ):
        landmark_xy = self.get_landmark_xy(dataset, preds, 3)

        plt.figure(figsize=(12, 8))
        for i in ids:
            y = preds[i]
            if i not in landmark_xy.keys():
                continue

            z = zs[i]
            x = [z * i for i in range(len(y))]

            x_shift = landmark_xy[i]["x"] * z
            y_shift = landmark_xy[i]["y"]
            x = np.array(x) - x_shift
            y = np.array(y) - y_shift
            plt.plot(x, y)

        plt.grid(True)
        plt.xlabel("z [mm]", fontsize=14)
        plt.ylabel("slice score", fontsize=14)
        plt.title(
            f"Slice scores for volumes in validation dataset\nAnchor: landmark {landmark_anchor} (0, 0)\n",
            fontsize=16,
        )
        if save_path:
            plt.savefig(
                save_path + f"slice-scores-{landmark_anchor}-anchor-xy-00.png",
                dpi=300,
                transparent=False,
                bbox_inches="tight",
            )
        plt.show()

    def plot_score2index_xfix(
        self,
        dataset,
        preds,
        zs,
        landmark_anchor=3,
        ids=np.arange(0, 20),
        save_path=False,
        title=None,
        colors=[],
    ):
        landmark_xy = self.get_landmark_xy(dataset, preds, landmark_anchor)
        plotted_ids = []
        if len(colors) == 0:
            colors = self.colors
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, idx in enumerate(ids):
            y = preds[idx]
            color = colors[i % len(colors)]
            if idx not in landmark_xy.keys():
                continue

            z = zs[idx]
            x = [(z * i) for i in range(len(y))]
            x_shift = landmark_xy[idx]["x"] * z
            x = np.array(x) - x_shift

            # get landmark positions
            x_landmarks, y_landmarks = self.landmark_positions(dataset, idx, x, y)

            plt.plot(x / 10, y, label=f"volume {idx}", color=color)
            plt.plot(x_landmarks/10, y_landmarks, color="black", marker=".", linestyle=" ")
            plotted_ids.append(idx)

        # self.set_scientific_style(ax) # TODO 
        plt.xlabel("z [cm]", fontsize=16)
        plt.ylabel("slice score", fontsize=16)
        plt.title(title, fontsize=16)
        if save_path:
            plt.savefig(
                save_path + f"slice-scores-{landmark_anchor}-anchor-x-0.png",
                dpi=300,
                transparent=False,
                bbox_inches="tight",
            )
        plt.show()

        return plotted_ids

    def plot_score2index(
        self,
        dataset,
        preds,
        zs,
        ids=np.arange(0, 20),
        save_path=False,
        colors=[],
        title=None,
    ):
        landmark_xy = self.get_landmark_xy(dataset, preds, 3)
        if len(colors) == 0:
            colors = self.colors

        fig, ax = plt.subplots(figsize=(12, 8))
        for i, idx in enumerate(ids):
            color = colors[i % len(colors)]
            if idx not in landmark_xy.keys():
                continue
            y = preds[idx]
            z = zs[idx]
            x = np.array([(z * i) for i in range(len(y))])

            # get landmark positions
            x_landmarks, y_landmarks = self.landmark_positions(dataset, idx, x, y)

            plt.plot(x / 10, y, label=f"volume {idx}", color=color)
            plt.plot(x_landmarks/10, y_landmarks, color="black", marker=".", linestyle=" ")

        # self.set_scientific_style(ax, legend_anchor=(0.98, 0.7)) # TODO 
        plt.xlabel("z [cm]", fontsize=16)
        plt.ylabel("slice score", fontsize=16)
        plt.title(title, fontsize=16)
        if save_path:
            plt.savefig(
                save_path + f"slice-scores.png",
                dpi=300,
                transparent=False,
                bbox_inches="tight",
            )
        plt.show()

    def landmark_positions(self, dataset, dataset_idx, x, y):
        landmarks = dataset.landmarks 
        landmark_dict_idx = [i for i in range(len(dataset)) if landmarks[i]["dataset_index"] == dataset_idx][0]
        indices = landmarks[landmark_dict_idx]["slice_indices"]
        x_landmarks = x[indices]
        y_landmarks = np.array(y)[indices]

        return x_landmarks, y_landmarks


    def predict_image(self, volume, indices=[]):
        if len(indices) == 0:
            inputx = torch.tensor(volume[:, np.newaxis, :, :]).cuda()
        else:
            inputx = torch.tensor(volume[indices, np.newaxis, :, :]).cuda()

        with torch.no_grad():
            self.model.eval()
            self.model.to("cuda")
            y = self.model(inputx)
            predicted_scores = [i.item() for i in y]

        return predicted_scores

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

    def plot_scores(self, vol_idx, save_path=False):
        (
            slice_idx,
            landmarks_idx,
            x,
            predicted_scores,
            expected_scores,
            errors,
        ) = self.landmarks2score(vol_idx, self.val_dataset, self.train_lm_summary)
        plt.figure(figsize=(15, 8))
        plt.plot(
            np.array(x),
            np.array(predicted_scores),
            linestyle="-",
            label="predicted slice score",
        )
        expected_f = interpolate.interp1d(slice_idx, expected_scores, kind="linear")
        plt.errorbar(
            slice_idx,
            expected_scores,
            yerr=errors,
            marker="x",
            color="orange",
            linestyle="",
            label="expected slice score (from training set)",
        )
        plt.plot(np.array(x), expected_f(np.array(x)), color="orange", linestyle="--")

        for i, landmark in enumerate(landmarks_idx):
            plt.annotate(
                landmark,
                xy=(
                    slice_idx[i] + 0.1,
                    expected_scores[i] + np.max(expected_scores) * 0.05,
                ),
            )

        plt.grid(True, axis="y")
        plt.legend(loc=0, fontsize=14)
        plt.title(f"Slice scores for volume {vol_idx}\n", fontsize=14)
        plt.xlabel("slice index", fontsize=14)
        plt.ylabel("slice score", fontsize=14)
        if save_path:
            plt.savefig(
                save_path + f"predicted-vs-estiamted-slice-score-vol{vol_idx}.png",
                dpi=300,
                transparent=False,
                bbox_inches="tight",
            )
        plt.show()
        torch.cuda.empty_cache()

    def plot_pred2expected_scores(self, save_path=False, ids=np.arange(0, 65)):
        plt.figure(figsize=(14, 8))
        for i in tqdm(ids):  # len(val_dataset)
            (
                slice_idx,
                landmarks_idx,
                x,
                predicted_scores,
                expected_scores,
                errors,
            ) = self.landmarks2score(i, self.val_dataset, self.train_lm_summary)
            if (len(np.unique(slice_idx)) != len(slice_idx)) or len(slice_idx) < 4:
                continue

            expected_f = interpolate.interp1d(slice_idx, expected_scores, kind="linear")
            label = (
                str(i)
                + "_"
                + self.val_dataset.landmarks[i]["filename"][0:8]
                + self.val_dataset.landmarks[i]["filename"][-10:]
                .replace(".npy", "")
                .replace("0", "")
            )
            plt.plot(expected_f(x), predicted_scores, label=label)

        plt.grid(True)

        xrange = np.arange(-7, 7)
        plt.plot(xrange, xrange, linestyle="--")
        plt.legend(loc=0)
        plt.xlabel("estimated slice score", fontsize=14)
        plt.ylabel("predicted slice score", fontsize=14)
        if save_path:
            plt.savefig(
                save_path + f"predicted-vs-estiamted-slice-score-multiple-volumes.png",
                dpi=300,
                transparent=False,
                bbox_inches="tight",
            )
        plt.show()

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

