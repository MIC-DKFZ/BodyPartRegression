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

    def mean_relative_deviation(self, landmark_preds, reference_results):
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
        mse_std = np.std(deviations) / np.sqrt(len(deviations))
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
        # self.landmark_names = names = ["pelvis-start", "pelvis-end", "kidney", "lung-start",
        #                               "liver-end", "lung-end", "teeth", "nose", "eyes-end"]
        # self.landmarkToClassMapping = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5}

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
        self.mse, self.mse_std = self.mean_relative_deviation(
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

