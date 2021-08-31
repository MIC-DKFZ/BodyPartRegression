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

import matplotlib.pyplot as plt
import numpy as np
import json, random
import pandas as pd

from bpreg.settings import ModelSettings
from bpreg.utils.training_utils import *


class Visualization:
    def __init__(self):
        pass

    def plot_landmarks(
        self,
        score_matrix: np.array,
        expected_scores: np.array,
        landmark_names: list,
        figsize: tuple = (16, 10),
        fontsize: float = 24,
        text_margin_top: float = 0.02,
        text_margin_right: float = 2,
        alpha: float = 0.7,
        ylim: None = None,
        colors: list = [],
    ):
        plt.figure(figsize=figsize)
        max_value = 0

        for idx in range(score_matrix.shape[1]):
            x = score_matrix[:, idx]
            if len(colors) == score_matrix.shape[1]:
                bins, _, _ = plt.hist(
                    x,
                    density=True,
                    alpha=alpha,
                    label=landmark_names[idx],
                    color=colors[idx],
                )
            else:
                bins, _, _ = plt.hist(
                    x,
                    density=True,
                    alpha=alpha,
                    label=landmark_names[idx],
                )

            mean = expected_scores[idx]
            plt.plot(
                [mean, mean],
                [0, 100],
                color="black",
                linestyle="--",
                linewidth=0.5,
            )

            plt.annotate(
                landmark_names[idx].replace("_", "-"),
                xy=(mean - text_margin_right, np.max(bins) + text_margin_top),
                fontsize=fontsize - 2,
            )

            if np.max(bins) > max_value:
                max_value = np.max(bins)

        if not ylim:
            plt.ylim((0, 1.1 * max_value))
        else:
            plt.ylim(ylim)
        plt.xticks(fontsize=fontsize - 2)
        plt.yticks(fontsize=fontsize - 2)
        plt.xlabel("Slice Scores", fontsize=fontsize)
        plt.ylabel("Relative Frequency", fontsize=fontsize)
        plt.xlim((-10, 110))


def grid_plot(
    X: np.ndarray,
    titles: list = [],
    cols: int = 4,
    rows: int = 4,
    save_path: str = "",
    cmap: str = "gray",
    vmin: int = 0,
    vmax: int = 250,
    figsize: tuple = (16, 10),
):
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


def plot_data(config: ModelSettings, kind: str = "train", cols=3, rows=3):
    df = get_dataframe(config)
    df = df[~df.pixel_spacingz.isna()]
    train_dataset, val_dataset, test_dataset = get_datasets(config, df)

    datasets = {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    dataset = datasets[kind]

    X = []
    titles = []
    for i in range(0, cols * rows):
        index = random.randint(0, len(dataset) - 1)
        titles.append(index)
        X.append(dataset[index][0][0, :, :])

    grid_plot(X, titles=titles, cols=cols, rows=rows, vmin=-1, vmax=1, figsize=(15, 15))


def plot_scores(filepath, save_path="", fontsize=18):
    def load_json(filepath):
        with open(filepath) as f:
            x = json.load(f)
        return x

    plt.figure(figsize=(12, 6))
    x = load_json(filepath)

    plt.plot(x["z"], x["cleaned slice scores"], label="Cleaned Slice Scores")
    plt.plot(
        x["z"],
        x["unprocessed slice scores"],
        label="Unprocessed Slice Scores",
        linestyle="--",
    )

    try:
        min_score = np.nanmin(x["unprocessed slice scores"])
        max_score = np.nanmax(x["unprocessed slice scores"])
        dflandmarks = pd.DataFrame(x["look-up table"]).T
        landmarks = dflandmarks[
            (dflandmarks["mean"] > min_score) & (dflandmarks["mean"] < max_score)
        ].sort_values(by="mean", ascending=False)
        for landmark, row in landmarks.iloc[[0, -1]].iterrows():
            plt.plot(
                [0, np.nanmax(x["z"])],
                [row["mean"], row["mean"]],
                linestyle=":",
                color="black",
                linewidth=0.8,
            )
            plt.text(
                5,
                row["mean"] + 1,
                landmark,
                fontsize=fontsize - 4,
                bbox=dict(
                    boxstyle="square",
                    fc=(1.0, 1, 1),
                ),
            )
    except:
        pass

    plt.xlabel("Height [mm]", fontsize=fontsize)
    plt.ylabel("Slice Scores", fontsize=fontsize)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.legend(loc=1, fontsize=fontsize)
    if np.nanmax(x["z"]) != 0:
        plt.xlim((0, np.nanmax(x["z"])))

    filename = filepath.split("/")[-1]
    if len(filename) > 60:
        plt.title(f"{filename[:60]}...\n", fontsize=fontsize - 2)
    else:
        plt.title(filename + "\n", fontsize=fontsize - 2)
    if len(save_path) > 0:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
