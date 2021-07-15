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
        plt.ylabel("Density Frequency Distribution", fontsize=fontsize)
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
