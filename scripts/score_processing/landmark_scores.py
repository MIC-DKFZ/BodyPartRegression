import sys, os
import numpy as np 
import pandas as pd
import torch 

sys.path.append("../../")
from scripts.dataset.base_dataset import get_full_volume_from_filepath, get_slices
from scripts.evaluation.normalized_mse import NormalizedMSE
from scripts.evaluation.accuracy import Accuracy
from src.settings.settings import * 
import json 

class LandmarkScores: 
    def __init__(self, 
                 data_path,
                 df, 
                 model, 
                 device="cuda",
                 drop_cols=["val", "train", "test"]): 
        
        self.data_path = data_path
        self.device = device
        self.model = model 
        
        # expect filenames to be with or without .npy ending
        self.filenames = [ f.replace(".npy", "") + ".npy" for f in df["filename"]]
        self.filepaths = [data_path + f for f in self.filenames]
        self.landmark_names = [ l for l in df.columns if not l in drop_cols + ["filename"]]

        self.index_matrix =  np.array(df.drop(["filename"] + drop_cols, axis=1, errors='ignore'))
        self.score_matrix = self.create_score_matrix()
        
        self.expected_scores = np.nanmean(self.score_matrix, axis=0)
        self.expected_scores_std = np.nanstd(self.score_matrix, axis=0)

        self.lookuptable = transform_lookuptable(self.create_lookuptable())
        
    def create_score_matrix(self): 
        with torch.no_grad(): 
            self.model.eval() 
            self.model.to(self.device)
            slice_score_matrix = np.full(self.index_matrix.shape, np.nan)
            for i in np.arange(len(self.index_matrix)): 
                filepath = self.filepaths[i]
                idxs = self.index_matrix[i, :]
                not_isnan = np.where(~np.isnan(idxs))
                idxs = idxs[not_isnan].astype(int)
                X = get_slices(filepath, idxs)
                y = self.model.predict_npy(X)
                slice_score_matrix[i, not_isnan] = y

        return slice_score_matrix
    
    def create_lookuptable(self): 
        lookuptable = {l: {} for l in self.landmark_names}

        for i, l in enumerate(self.landmark_names): 
            lookuptable[l]["mean"] = self.expected_scores[i]
            lookuptable[l]["std"] = self.expected_scores_std[i]
        return lookuptable
    
    
    def print_lookuptable(self): 
        for landmark, values in self.lookuptable.items(): 
            mean = np.round(values["mean"], 3)
            std = np.round(values["std"], 3)
            print(f"{landmark:<15}:\t {mean}+-{std}")    

    def save_lookuptable(self, filepath): # TODO 
        with open(filepath, "w") as f:
            json.dump(self.lookuptable, f)

            
class LandmarkScoreBundle: 
    def __init__(self, data_path, landmark_path, model): 
        df_database = pd.read_excel(landmark_path, sheet_name="database")
        df_train = pd.read_excel(landmark_path, sheet_name="landmarks-train")
        df_val = pd.read_excel(landmark_path, sheet_name="landmarks-val")
        df_test = pd.read_excel(landmark_path, sheet_name="landmarks-test")

        self.landmark_scores = {
            "validation":  LandmarkScores(data_path, df_val, model), 
            "train":  LandmarkScores(data_path, df_train, model),  
            "test":  LandmarkScores(data_path, df_test, model), 
            "train+val":  LandmarkScores(data_path, df_database, model), 
        }
        self.nmse = NormalizedMSE()
    def nMSE(self, target="validation", reference="train"): 
        nmse, nmse_std, _ = self.nmse.from_matrices(
                                self.landmark_scores[target].score_matrix, 
                                self.landmark_scores[reference].score_matrix)
        return nmse, nmse_std
    
    def accuracy(self, target_dataset, reference="train", class2landmark=CLASS_TO_LANDMARK_5): 
        acc = Accuracy(self.landmark_scores[reference].expected_scores, class2landmark)
        myAccuracy = acc.from_dataset(self.model, target_dataset)
        return myAccuracy 
    
    def nMSE_per_landmark(self, target="validation", reference="train"): 
        pass
    



def transform_0to100(score, lookuptable, min_value=np.nan): 
    if np.isnan(min_value): min_value = lookuptable["pelvis_start"]["mean"]
    max_value = lookuptable["eyes_end"]["mean"]

    score = score - min_value
    score = score * 100 / (max_value - min_value)

    return score


def transform_lookuptable(lookuptable):
    lookup_copy = {key: {} for key in lookuptable}
    for key in lookuptable:
        lookup_copy[key]["mean"] = np.round(
            transform_0to100(lookuptable[key]["mean"], lookuptable), 3
        )
        lookup_copy[key]["std"] = np.round(
            transform_0to100(lookuptable[key]["std"], lookuptable, min_value=0.0), 3
        )

    return lookup_copy
