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
from src.settings.settings import *

def grid_plot(X: np.ndarray, 
             titles: list = [], 
             cols: int = 4, 
             rows: int = 4, 
             save_path: str = "", 
             cmap: str = "gray", 
             vmin: int = 0,
             vmax: int = 250, 
             figsize: tuple = (16,10)):
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    idx = 0
    for row in range(rows):
        for col in range(cols):
            axs[row, col].imshow(X[idx], cmap=cmap, vmin=vmin, vmax=vmax)
            if len(titles) == cols * rows:
                axs[row, col].set_title(titles[idx], fontsize=15)
            axs[row, col].set_yticklabels([])
            axs[row, col].set_xticklabels([])
            idx += 1

    if len(save_path) > 0:
        plt.savefig(save_path)
    plt.show()
    
class Evaluation:
    """General evaluation of slice score predictions.

    Args:
        fontsize: fontsize for plots
    """

    def __init__(self, fontsize: int = 16):
        self.landmark_names = LANDMARK_NAMES
        self.landmarkToClassMapping = LANDMARK_CLASS_MAPPING
        self.colors = COLORS
        self.fontsize = fontsize

    def plot_landmarks(
        self,
        val_landmark_preds: dict,
        min_value: float = -20.0,
        max_value: float = 20.0,
        save_path: str = "",
        bin_width: float = 0.5,
        title: str = None,
    ):
        """Plot for each landmark the predicted slice score distribution as histogram.

        Args:
            val_landmark_preds (dict): contains for each landmark the slice score predictions.
            min_value (float, optional): min value for x-axis. Defaults to -20.0.
            max_value (float, optional): max value for x-axis. Defaults to 20.0.
            save_path (str, optional): save path for figure. Defaults to "".
            bin_width (float, optional): bin width of histograms. Defaults to 0.5.
            title (str, optional): title of figure. Defaults to None.
        """

        _, ax = plt.subplots(figsize=(18, 12))
        for landmark, values in val_landmark_preds.items():
            if len(values) <= 1:
                continue
            x = plt.hist(
                values,
                density=True,
                label=self.landmark_names[landmark],
                bins=np.arange(min(values), max(values) + bin_width, bin_width),
                alpha=0.75,
            )
            plt.annotate(
                self.landmark_names[landmark],
                xy=(np.median(values) - 0.5, np.max(x[0]) + 0.04),
                fontsize=self.fontsize - 2,
            )
            plt.xlabel("slice score", fontsize=self.fontsize)
            plt.ylabel("frequency distribution", fontsize=self.fontsize)

        self.set_scientific_style(
            ax,
            legend=False,
            minor_xticks=0.25,
            major_xticks=1,
            minor_yticks=0.1,
            major_yticks=0.5,
        )

        plt.grid(True, axis="x")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim((min_value, max_value))
        plt.title(title, fontsize=18)
        if len(save_path) > 0:
            plt.savefig(
                save_path + "landmark-prediction-historgram.png",
                dpi=300,
                transparent=False,
                bbox_inches="tight",
            )

    def get_landmark_prediction(self, dataset, dataset_preds: dict):
        """
        return {landmark: [landmark-scores] for landmark in landmarks} dictionary
        """
        landmark_preds = {i: [] for i in range(0, 12)}
        landmark_preds_ids = {i: [] for i in range(0, 12)}
        for i, myDict in dataset.landmarks.items():
            index = myDict["dataset_index"]
            if index not in dataset_preds.keys():
                continue
            ys = dataset_preds[index]
            for landmark in myDict["defined_landmarks_i"]:
                lm_idx = myDict["slice_indices"][
                    np.where(myDict["defined_landmarks_i"] == landmark)[0][0]
                ]
                lm_y = ys[lm_idx]
                landmark_preds[landmark].append(lm_y)
                landmark_preds_ids[landmark].append(index)
        return landmark_preds, landmark_preds_ids

    def plot_similar_images(
        self,
        dataset,
        preds,
        min_value=-20,
        max_value=20,
        save_path=False,
        skip_diff=0.25,
        vmin=-1, 
        vmax=1
    ):
        targets = np.linspace(min_value, max_value, 5, dtype=int)

        for target in targets:
            print(f"Plot corresponding images to the target score {target}")
            if save_path:
                save_path2 = save_path + f"similar-images-to-score-{target}.png"
            else:
                save_path2 = False
            self.plot_closest_slices(
                dataset, preds, target, skip_diff=skip_diff, save_path=save_path2, vmin=vmin, vmax=vmax
            )

    def plot_closest_slices(
        self, dataset, predict_dict, y_target, skip_idx=[], skip_diff=3, save_path=False,
        vmin=-1, vmax=1,
    ):
        vols = []
        titles = []
        vol_indices = np.array(list(predict_dict.keys()))
        np.random.shuffle(vol_indices)

        for vol_idx in vol_indices:
            if vol_idx in skip_idx:
                continue
            closest_slice, closest_value, _ = self.get_closest_slice(
                dataset, predict_dict, y_target, vol_idx
            )
            if np.abs(closest_value - y_target) < skip_diff:
                vols.append(closest_slice)
                titles.append(f"Volume {vol_idx} closest image to score {y_target}")
        grid_plot(vols[0:9], titles[0:9], cols=3, rows=3, save_path=save_path, vmin=vmin, vmax=vmax)

    def get_closest_slice(self, dataset, predict_dict, y_target, vol_idx):
        y_pred = np.array(predict_dict[vol_idx])
        min_idx = np.argmin(np.abs(y_pred - y_target)).item()
        try:  # TODO!
            vol = dataset.preprocessed_volumes[vol_idx]
        except:
            vol = dataset.get_full_volume(vol_idx)
        if len(vol.shape) == 4:
            closest_slice = vol[min_idx, 0, :, :]
        else:
            closest_slice = vol[min_idx, :, :]
        closest_value = y_pred[min_idx]
        return closest_slice, closest_value, min_idx

    def set_scientific_style(
        self,
        ax,
        legend_anchor=(0.2, 0.98),
        minor_xticks=5,
        major_xticks=20,
        minor_yticks=0.5,
        major_yticks=2,
        legend=True,
        labelsize=12,
    ):
        plt.rc("font", family="serif")
        plt.grid(True)
        if legend:
            plt.legend(bbox_to_anchor=legend_anchor, loc=1, frameon=False, fontsize=12)

        ax.xaxis.set_tick_params(
            which="major", size=8, width=1, direction="in", top="on"
        )
        ax.xaxis.set_tick_params(
            which="minor", size=5, width=1, direction="in", top="on"
        )
        ax.yaxis.set_tick_params(
            which="major", size=8, width=1, direction="in", right="on"
        )
        ax.yaxis.set_tick_params(
            which="minor", size=5, width=1, direction="in", right="on"
        )

        ax.xaxis.set_minor_locator(mlp.ticker.MultipleLocator(minor_xticks))
        ax.xaxis.set_major_locator(mlp.ticker.MultipleLocator(major_xticks))

        ax.yaxis.set_minor_locator(mlp.ticker.MultipleLocator(minor_yticks))
        ax.yaxis.set_major_locator(mlp.ticker.MultipleLocator(major_yticks))

        ax.tick_params(axis="both", which="major", labelsize=labelsize)

    def get_landmark_prediction_summary(self, dataset, model):
        """
        return {landmark: {lm: landmark-metric, mean: mean-prediction, var: variance-of-prediction}
        for landmark in landmarks}
        """
        with torch.no_grad():
            model.eval()
            model.to("cuda")
            landmark_prediction = np.full(
                (len(dataset.landmarks), len(dataset.landmark_names)), np.nan
            )
            for i, landmark_dir in dataset.landmarks.items():
                x = dataset.get_full_volume(landmark_dir["dataset_index"])
                x = torch.tensor(x[landmark_dir["slice_indices"], :, :])[
                    :, np.newaxis, :, :
                ]
                # calculate prediction
                ys = self.model(x.cuda())
                ys = np.array([y.item() for y in ys])

                landmark_prediction[(i, landmark_dir["defined_landmarks_i"])] = ys

        landmark_vars = np.nanvar(landmark_prediction, axis=0)
        total_var = np.nanvar(landmark_prediction)
        landmark_predictions = np.nanmean(landmark_prediction, axis=0)
        results = {
            i: {
                "lm": np.round(landmark_vars[i] / total_var, 4),
                "mean": np.round(landmark_predictions[i], 3),
                "var": np.round(landmark_vars[i], 3),
            }
            for i in range(len(landmark_vars))
        }
        return results

    def landmark_dict_summary(self, landmark_predictions):
        total_predictions = []
        for myList in landmark_predictions.values():
            total_predictions += myList
        total_var = np.nanvar(total_predictions)

        myDict = {
            key: {
                "lm": np.round(np.var(landmark_predictions[key]) / total_var, 4),
                "mean": np.round(np.mean(landmark_predictions[key]), 2),
                "var": np.round(np.var(landmark_predictions[key]), 2),
            }
            for key in landmark_predictions.keys()
            if not len(landmark_predictions[key]) == 0
        }

        return myDict

    def normalized_mse(self, landmark_preds, reference_results):        
        # TODO -- umschreiben -- siehe bpr_model -> diese Funktion nutzen     
        deviations = []

        # normalize deviations
        max_value = reference_results[8]["mean"]  # pelvis-start landmark
        min_value = reference_results[0]["mean"]  # eyes-end landmark

        for landmark in landmark_preds.keys():
            if not landmark in reference_results.keys():
                continue
            landmark_deviations = (
                (np.array(landmark_preds[landmark]) - reference_results[landmark]["mean"])
                / (max_value - min_value)
            ) ** 2
            deviations += list(landmark_deviations)

        mse = np.mean(deviations)
        mse_std = np.std(deviations) / np.sqrt(len(deviations)) # TODO Fehler !!! 
        return mse, mse_std

    def accuracy(self, dataset, predictions, reference_results, reverse=False):
        preds = []
        obs = []
        s = []
        for key, landmark_dict in dataset.landmarks.items():
            obs_classes, pred_classes, scores, volume_accuracy, _ = self.classify_volume(
                landmark_dict, predictions, reference_results, reverse=reverse
            )
            preds += list(pred_classes)
            obs += list(obs_classes)
            s += list(scores)

        if len(np.array(obs)) != len(np.array(preds)):
            print(len(np.array(obs)), len(np.array(preds)))
            print(obs)
            print(preds)
            raise ValueError("Unequal lengths. Can't compute the accuracy. ")

        evaluation = (np.array(obs) == np.array(preds)) * 1

        acc = np.sum(evaluation) / len(obs)
        std = np.std(evaluation) / np.sqrt(len(obs))
        return acc, std

    def predict_dataset(self, dataset, model):
        preds = {}
        indices = np.arange(len(dataset))
        indices = indices[0:75]
        zs = {}

        for i in tqdm(indices):
            volume = dataset.get_full_volume(i)
            zs[i] = dataset.z_spacings[i]
            volume = torch.tensor(volume[:, np.newaxis, :, :])
            with torch.no_grad():
                model.eval()
                model.to("cuda")
                try:
                    ys = model(volume.cuda())
                except:
                    print("Out of memory", volume.shape)
                    continue
                preds[i] = [y.item() for y in ys]
        return preds, zs

    def classify_volume(
        self, landmark_dict, predictions, reference_results, reverse=False
    ):
        vol_idx = landmark_dict["dataset_index"]
        if not vol_idx in predictions.keys():
            return [], [], [], 0, None
        analyzed_indices, obs_classes = self.observed_classes(landmark_dict)
        scores = np.array(predictions[vol_idx])[analyzed_indices]
        pred_classes = np.array(
            [self.score_to_class(y, reference_results, reverse=reverse) for y in scores]
        )
        accuracy = np.sum((obs_classes == pred_classes) * 1) / len(pred_classes)
        return obs_classes, pred_classes, scores, accuracy, analyzed_indices

    def observed_classes(self, myDict):
        analyzed_indices = np.arange(
            min(myDict["slice_indices"]), max(myDict["slice_indices"])
        )
        start_slice = min(myDict["slice_indices"])
        obs_classes = []
        for landmark in myDict["defined_landmarks_i"]:
            index = int(np.where(myDict["defined_landmarks_i"] == landmark)[0])
            stop_slice = myDict["slice_indices"][index]
            myClass = self.landmarkToClassMapping[landmark]
            if (stop_slice - start_slice) < 0:
                continue
            obs_classes += [myClass] * (stop_slice - start_slice)
            start_slice = stop_slice

        return analyzed_indices, np.array(obs_classes)

    def score_to_class(self, score, results, reverse):
        if not reverse:
            for l_idx, map_value in  self.landmarkToClassMapping.items(): 
                upper_condition = (score < results[l_idx]["mean"])
                lower_condition = 1
                if l_idx > 0: 
                    lower_condition = (score > results[l_idx - 1]["mean"])
                if (lower_condition & upper_condition): 
                    return map_value

        else:
            for l_idx, map_value in  self.landmarkToClassMapping.items(): 
                upper_condition = (score > results[l_idx]["mean"])
                lower_condition = 1
                if l_idx > 0: 
                    lower_condition = (score < results[l_idx - 1]["mean"])
                if (lower_condition & upper_condition): 
                    return map_value


    def plot_slope_histogramm(
        self, filename="slope-histogramm-2.png", bin_width=0.00025
    ):
        slope_path = "/home/AD/s429r/Documents/Data/Results/misleading-slopes/"
        save_path = slope_path + filename
        mySlopes = np.load(slope_path + "slopes.npy")
        _, ax = plt.subplots(figsize=(18, 12))
        plt.hist(mySlopes, bins=np.arange(-0.015, 0.02 + bin_width, bin_width))
        self.set_scientific_style(
            ax, major_xticks=0.005, minor_xticks=0.0025, labelsize=14
        )
        plt.yscale("log")
        plt.xlabel("Slope [1/mm]", fontsize=18)
        plt.ylabel("Counts", fontsize=18)
        plt.savefig(save_path, dpi=300, transparent=False, bbox_inches="tight")
        plt.show()
