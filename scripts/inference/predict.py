import sys, os, datetime
import argparse,  pickle, json
import torch
import pytorch_lightning as pl
from sklearn.linear_model import RANSACRegressor
from scipy.ndimage import gaussian_filter
import numpy as np 


sys.path.append("../../")

from scripts.postprocessing.datasanitychecks import DataSanityCheck
from scripts.postprocessing.slicescoreprocessing import SliceScoreProcessing
from scripts.postprocessing.lookuptable import LookUpTable
from scripts.preprocessing.nifti2npy import Nifti2Npy
from scripts.network_architecture.bpr_model import BodyPartRegression 



class Predict(SliceScoreProcessing): 
    """
    run analysis: 
        - slice score preprocessing
        - slice score postprocessing
        - data sanity checks 
        - caluclate 
        
    todo add inputs: 
    - reference table as input
    - mean and std of slice score slope
    """
    def __init__(self, base_dir, slope_mean, slope_std, lower_bound_score, upper_bound_score, smoothing_sigma,
                 lookup_table={}): 
        SliceScoreProcessing.__init__(self, base_dir)

        self.slope_mean = slope_mean 
        self.slope_std = slope_std 
        
        self.lower_bound_score = lower_bound_score
        self.upper_bound_score = upper_bound_score
        self.smoothing_sigma = smoothing_sigma
        self.n2n = Nifti2Npy(target_pixel_spacing=3.5, 
               min_hu=-1000,
               max_hu=1500, 
               size=128) 
        
        self.lookup_table=lookup_table

    def load_model(self, base_dir): 
        self.base_filepath = base_dir
        self.config_filepath = base_dir + "config.p"
        self.model_filepath = base_dir + "model.pt"

        with open(self.config_filepath, "rb") as f: 
            self.config = pickle.load(f)
            
        self.model = BodyPartRegression(alpha=self.config["alpha"], lr=self.config["lr"], base_model=self.config["base_model"])
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()
        self.model.to("cuda")
        
    def body_range(self, slice_scores, lower_slice_index, upper_bound_score): 
        body_region = np.where(slice_scores < upper_bound_score)[0]
        if len(body_region) == 0: 
            return np.array([]), 0
        upper_slice_index = np.max(body_region) + 1
        return np.arange(lower_slice_index, upper_slice_index), upper_slice_index

    def get_body_part_examined(self, slice_scores): 
        legs_array, legs_max       = self.body_range(slice_scores, 0, self.lookup_table[0]["mean"])
        pelvis_array, pelvis_max   = self.body_range(slice_scores, legs_max, self.lookup_table[1]["mean"])
        abdomen_array, abdomen_max = self.body_range(slice_scores, pelvis_max, self.lookup_table[3]["mean"])
        lung_array, lung_max       = self.body_range(slice_scores, abdomen_max, self.lookup_table[5]["mean"])
        neck_array, neck_max       = self.body_range(slice_scores, lung_max, self.lookup_table[6]["mean"])
        head_array = np.arange(neck_max, len(slice_scores))

        body_part_examined = {"legs": list(legs_array.astype(float)), 
                             "pelvis": list(pelvis_array.astype(float)), 
                             "abdomen": list(abdomen_array.astype(float)), 
                             "lungs": list(lung_array.astype(float)), 
                             "shoulder-neck": list(neck_array.astype(float)), 
                             "head": list(head_array.astype(float))}
        
        return body_part_examined 
    
    def predict_scores(self, nifti_path): 
        # get nifti file as tensor
        x, pixel_spacings = self.n2n.preprocess_nifti(nifti_path)    
        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x).cuda()
        
        # predict slice-scores 
        scores = self.predict_tensor(x_tensor)
        
        return scores, pixel_spacings
    
    def transform_0_100(self, score, min_value=False): 
        if isinstance(min_value, bool): min_value = self.lookup_table[0]["mean"]
        max_value = self.lookup_table[8]["mean"]

        score = score - min_value
        score = score * 100/(max_value - min_value)
        
        return score
    
    def transform_lookup(self): 
        lookup_copy = {key: {} for key in self.lookup_table}
        for key in self.lookup_table: 
            lookup_copy[key]["landmark-name"] = self.lookup_table[key]["landmark-name"]
            lookup_copy[key]["mean"] = np.round(self.transform_0_100(self.lookup_table[key]["mean"]), 3)
            lookup_copy[key]["std"]  = np.round(self.transform_0_100(self.lookup_table[key]["std"], min_value=0), 3)
        
        return lookup_copy
    
    def predict(self, nifti_path, output_path): 
        # Scores vorhersagen
        scores, pixel_spacings = self.predict_scores(nifti_path)
        
        # Validierung
        dsc = DataSanityCheck(scores, 
                              pixel_spacings[2], 
                              self.slope_mean, 
                              self.slope_std, 
                              self.lower_bound_score, 
                              self.upper_bound_score,
                              self.smoothing_sigma)
        cleaned_scores, valid_indices = dsc.remove_invalid_regions(scores)
        reverse_zordering = dsc.is_reverse_zordering()
        valid_zspacing = dsc.is_valid_zspacing()
        
        # Estimate BodyPartExamined
        body_part_examined = self.get_body_part_examined(cleaned_scores)
        
        # transform to 0-100
        cleaned_scores = self.transform_0_100(cleaned_scores)
        scores = self.transform_0_100(scores)
        lookup_transformed = self.transform_lookup()
        
        # write output json file
        json_output = {
                      "slice scores": list(cleaned_scores), 
                      "valid indices": list(valid_indices.astype(float)), 
                      "unprocessed slice scores": list(scores), 
                      "body part examined": body_part_examined,
                      "look-up table": lookup_transformed, 
                      "reverse z-ordering": reverse_zordering, 
                      "valid z-spacing": valid_zspacing, 
                      "z-spacing": np.round(float(pixel_spacings[2]), 2), 
                      "expected z-spacig": np.round(dsc.zhat, 2)
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_output, f)
        
        return json_output


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", default="")
    parser.add_argument("--o", default="")

    value = parser.parse_args()
    ipath = value.i
    opath = value.o

    base_dir = "../../src/models/loh-ldist-l2/sigma-dataset-v11/"
    sys.path.append("../../../s429r/") # TODO!! Modell indieser Ordnerstruktur erzeugen und ausführbar machen -> Code auf GPU laden 
    # TODO Model in Dockerfile bekommen --> nicht Pfad außerhalb nutzen 
    slope_mean = 0.012527 # TODO 
    slope_std = 0.00138 # TODO 
    lower_bound_score = -6
    upper_bound_score = 6
    smoothing_sigma = 10 # TODO 

    # import lookup-table
    with open(base_dir + "lookuptable.json", "r") as f: 
        table = json.load(f)
        # convert to integer keys
        table = {int(key): table[key] for key in table.keys()}

    model = Predict(base_dir, 
                  slope_mean, 
                  slope_std, 
                  lower_bound_score, 
                  upper_bound_score, 
                  smoothing_sigma,
                  lookup_table=table)

    nifti_path = "../../data/Task049_StructSeg2019_Task1_HaN_OAR_20_0000.nii.gz"
    output_path = "../../data/Task049_StructSeg2019_Task1_HaN_OAR_20_0000.json"

    model.predict(nifti_path, output_path)