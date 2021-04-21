
import numpy as np 
import torch
import os, sys

sys.path.append("../")
from scripts.inference.predict_volume import PredictVolume


class LookUpTable(PredictVolume): 
    def __init__(self, base_dir): 
        PredictVolume.__init__(self, base_dir)
        self.description = {0: {"landmark-name": "pelvis_start"}, 
                             1: {"landmark-name": "pelvis_end"}, 
                             2: {"landmark-name": "kidneys"}, 
                             3: {"landmark-name": "lung_start"}, 
                             4: {"landmark-name": "liver_end"}, 
                             5: {"landmark-name": "lung_end"}, 
                             6: {"landmark-name": "teeths"}, 
                             7: {"landmark-name": "nose"}, 
                             8: {"landmark-name": "eyes_end"}}
        
    def get_lookup_table(self, dataset):
        """
        return {landmark: {lm: landmark-metric, mean: mean-prediction, var: variance-of-prediction} 
        for landmark in landmarks}
        """
        with torch.no_grad(): 
            self.model.eval()
            self.model.to("cuda")
            landmark_prediction = np.full((len(dataset.landmarks), len(dataset.landmark_names)), np.nan)
            for i, landmark_dir in dataset.landmarks.items(): 
                x = dataset.get_full_volume(landmark_dir["dataset_index"])
                x = torch.tensor(x[landmark_dir["slice_indices"], :, :])[:, np.newaxis, :, :]
                # calculate prediction
                ys = self.model(x.cuda())
                ys = np.array([y.item() for y in ys])

                landmark_prediction[(i, landmark_dir["defined_landmarks_i"])] = ys

        landmark_vars = np.nanvar(landmark_prediction, axis=0)
        landmark_stds = np.nanstd(landmark_prediction, axis=0)
        total_var = np.nanvar(landmark_prediction)
        landmark_predictions = np.nanmean(landmark_prediction, axis=0)
        results = {i: {"landmark-name": self.description[i]["landmark-name"],
                       "mean": np.round(landmark_predictions[i],3), 
                       "std": np.round(landmark_stds[i],3)} for i in range(len(landmark_vars))}
                       #"landmark metric": np.round(landmark_vars[i]/total_var, 4)}
        return results 
    
    
    def print(self, lookup_table): 
        for i in lookup_table.keys():
            name = lookup_table[i]["landmark-name"] # {lookup_table[i]['lm'] :1.4f}\t" + 
            print(f"{i} {name:<15}\t" +
                  f"{lookup_table[i]['mean']:<6}" \
                  f" +- {lookup_table[i]['std']}")
