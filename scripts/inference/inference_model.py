import numpy as np 
import os, sys
import torch 
import json, pickle

sys.path.append("../../")
from scripts.score_processing.datasanitychecks import DataSanityCheck
from scripts.score_processing.bodypartexamined import BodyPartExamined
from scripts.preprocessing.nifti2npy import Nifti2Npy
from scripts.network_architecture.bpr_model import BodyPartRegression
from scripts.score_processing.scores import Scores

# TODO Use InferenceModel in modelEvaluation 
# TODO put prediction code from bpr_model --> InferenceModel 
# TODO convert Dockercontainer again and starter.py
# TODO remove predict.py
# TODO Use InferenceModel for SliceScoreProcessing 
# TODO Write Tests for scores2bodyrange 
# TODO Include data sanity checks 
# TODO BodyPartExamined hinzufügen 
# TODO DataSanityChecks: expected z-spacing

class InferenceModel: 
    """
    Body Part Regression Model for inference purposes. 
    
    Args:
        base_dir (str]): Path which includes model related file. 
        Structure of base_dir: 
        base_dir/ 
            model.pt - includes model 
            settings.json - includes mean slope and mean slope std
            lookuptable.json - includes lookuptable as reference 
        device (str, optional): [description]. "cuda" or "cpu" 
    """
    def __init__(self, base_dir, device="cuda"): 

        self.base_dir = base_dir
        self.device = device

        self.model = load_model(base_dir, device=self.device)
        self.load_lookuptable()
        self.load_settings()
        self.n2n = Nifti2Npy(
            target_pixel_spacing=3.5, min_hu=-1000, max_hu=1500, size=128
        )

    def load_lookuptable(self): 
        path = self.base_dir + "lookuptable.json"
        if not os.path.exists: 
            return np.nan 

        with open(path, "rb") as f: 
            lookuptable = json.load(f)

        self.lookuptable_original = lookuptable["original"]
        self.lookuptable = lookuptable["transformed"]

    def load_settings(self): 
        path = self.base_dir + "settings.json"
        if not os.path.exists: return np.nan 

        with open(path, "rb") as f: 
            mySettings = json.load(f)

        self.slope_mean = mySettings["slope_mean"]
        self.slope_std = mySettings["slope_std"]

    def predict_tensor(self, tensor, n_splits=200): 
        scores = []
        n = tensor.shape[0]
        slice_splits = list(np.arange(0, n, n_splits)) 
        slice_splits.append(n)

        with torch.no_grad(): 
            self.model.eval() 
            self.model.to(self.device)
            for i in range(len(slice_splits) - 1): 
                min_index = slice_splits[i]
                max_index = slice_splits[i+1]
                score = self.model(tensor[min_index:max_index,:, :, :].to(self.device))
                scores += [s.item() for s in score]

        scores = np.array(scores)
        return scores

    def predict_npy(self, x, n_splits=200): 
        x_tensor = torch.tensor(x[:, np.newaxis, :, :]).to(self.device)
        scores = self.predict_tensor(x_tensor, n_splits=n_splits)
        return scores

    def predict_nifti(self, nifti_path): 
        # get nifti file as tensor
        x, pixel_spacings = self.n2n.preprocess_nifti(nifti_path)
        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x)
        x_tensor.to(self.device)

        # predict slice-scores
        scores = self.predict_tensor(x_tensor)
        return scores, pixel_spacings      

    def datasanitychecks(self, scores, z_spacing): 
        # validation
        dsc = DataSanityCheck(
            scores,
            z_spacing, 
            self.slope_mean,
            self.slope_std,
            self.lower_bound_score,
            self.upper_bound_score,
            self.smoothing_sigma,
        )
        cleaned_scores, valid_indices = dsc.remove_invalid_regions(scores)
        reverse_zordering = dsc.is_reverse_zordering()
        valid_zspacing = dsc.is_valid_zspacing()

        return {
            "slice scores": list(cleaned_scores), 
            "valid indices": list(valid_indices.astype(float)), 
            "unprocessed slice scores": list(scores), 
            "reverse z-ordering": reverse_zordering, 
            "valid z-spacing": valid_zspacing, 
            "z-spacing": np.round(float(z_spacing), 2), 
            "expected z-spacing": np.round(float(dsc.z_hat), 2)
        }

    def nifti2json(self, nifti_path, output_path): 
        slice_score_values, pixel_spacings = self.predict_nifti(nifti_path)
        slice_scores = Scores(slice_score_values, 
                              pixel_spacings[2], 
                              transform_min = self.lookuptable_original["pelvis_start"]["mean"], 
                              transform_max = self.lookuptable_original["eyes_end"]["mean"])

        output = {"slice scores": list(slice_scores.values.astype(np.float64)), 
            "z": list(slice_scores.z.astype(np.float64)), 
            "valid z": list(slice_scores.valid_z.astype(np.float64)), 
            "unprocessed slice scores": list(slice_scores.original_transformed_values.astype(np.float64)), 
            "look-up table": self.lookuptable, 
            "z-spacing": pixel_spacings[2].astype(np.float64)}

        if len(output_path) == 0: return output

        with open(output_path, "w") as f: 
            json.dump(output, f)

        return output

def load_model(base_dir, 
                model_file="model.pt", 
                config_file="config.p", 
                device="cuda"): 
    config_filepath = base_dir + config_file # TODO
    model_filepath = base_dir + model_file # TODO verallgemeinern

    with open(config_filepath, "rb") as f: 
        config = pickle.load(f)
        
    model = BodyPartRegression(alpha=config["alpha"], 
                                    lr=config["lr"]) 
    model.load_state_dict(torch.load(model_filepath))
    model.eval()
    model.to(device)

    return model

# TODO 
"""
json_output = {
"slice scores": list(cleaned_scores),
"valid indices": list(valid_indices.astype(float)),
"unprocessed slice scores": list(scores),
"body part examined": body_part_examined,
"look-up table": lookup_transformed,
"reverse z-ordering": reverse_zordering,
"valid z-spacing": valid_zspacing,
"z-spacing": np.round(float(pixel_spacings[2]), 2),
"expected z-spacig": np.round(dsc.zhat, 2),
}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", default="")
    parser.add_argument("--o", default="")
    parser.add_argument("--g", default=1) 

    value = parser.parse_args()
    ipath = value.i
    opath = value.o
    gpu = value.g

    base_dir = "../../src/models/loh-ldist-l2/sigma-dataset-v11/"
    model = Predict(
        base_dir,
        smoothing_sigma=10,
        gpu=gpu
    )

    data_path = "../../data/test_cases/"
    nifti_paths = [data_path + f for f in os.listdir(data_path) if f.endswith(".nii.gz")]
    for nifti_path in tqdm(nifti_paths): 
        output_path = nifti_path.replace("test_cases", "test_results").replace(".nii.gz", ".json")
        model.predict(nifti_path, output_path)
    
