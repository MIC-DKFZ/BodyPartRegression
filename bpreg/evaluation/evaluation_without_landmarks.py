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

from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np

from bpreg.score_processing import Scores
from bpreg.utils.linear_transformations import *
from bpreg.dataset.base_dataset import BaseDataset


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    cmap =  plt.cm.get_cmap(name, n)
    return [cmap(i) for i in np.arange(n)]


class ValidationVolume:
    def __init__(
        self,
        inference_model,
        val_dataset: BaseDataset,
        idx: int,
        fontsize: float = 22,
    ):
        self.data_idx = idx
        if not isinstance(self.data_idx, list):
            self.data_idx = [self.data_idx]

        self.z_array_list = []
        self.scores_list = []
        self.filename_list = []
        self.color_list = get_cmap(len(self.data_idx))
        for i in range(len(self.data_idx)):
            filename = val_dataset.filenames[self.data_idx[i]]
            X = val_dataset.get_full_volume(self.data_idx[i])
            z = val_dataset.z_spacings[self.data_idx[i]]
            scores = inference_model.predict_npy_array(X)

            z_array = np.arange(0, len(X)) * z

            self.z_array_list.append(z_array)
            self.scores_list.append(scores)
            self.filename_list.append(filename)

        self.fontsize = fontsize
            

    def plot_scores(self, set_figsize=(14,8), legend=None, postprocess_volumes=False):
        if set_figsize:
            plt.figure(figsize=set_figsize)
        
        if postprocess_volumes:
            self.merge_scores_for_volume()
            for i in range(len(self.data_idx)):
                # plot scores for volume
                plt.plot(
                    self.z_array_list[i],
                    self.scores_list[i],
                    color=self.volume_color_list[i],
                    linewidth=1,
                    label=self.volumename_list[i]
                )

        else:
            for i in range(len(self.data_idx)):
                # plot scores for volume
                plt.plot(
                    self.z_array_list[i],
                    self.scores_list[i],
                    color=self.color_list[i],
                    linewidth=1,
                    label=self.filename_list[i]
                )

        plt.xlabel("z [mm]", fontsize=self.fontsize)
        plt.ylabel("Slice Score", fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize - 2)
        plt.yticks(fontsize=self.fontsize - 2)
        if legend:
            plt.legend(loc=0)


    def merge_scores_for_volume(self):
        """merge scores for volume parts belonging to same patient
        """
        self.volumename_list = [filename.split(".")[0] for filename in self.filename_list]
        unique_volume_list = list(np.unique(self.volumename_list))
        volume_colors = self.color_list[0:len(unique_volume_list)]
        self.volume_color_list = []
        for volumename in self.volumename_list:
            i = unique_volume_list.index(volumename)
            self.volume_color_list.append(volume_colors[i])# ToDo: LinearSegmentedColorObject
        # Reset z_array_list for volume
        