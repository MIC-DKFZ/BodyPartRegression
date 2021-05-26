import sys, os
import argparse, json
import torch
from tqdm import tqdm
import numpy as np


sys.path.append("../../")

from scripts.postprocessing.datasanitychecks import DataSanityCheck
from scripts.preprocessing.nifti2npy import Nifti2Npy

class Predict:
    """Get body part examined estimate for CT volume.
    Predict slice scores for a given 3D CT nifti-file.
    Get information about the seen bodypart in the CT file based on the look up table,
    which maps landmarks to slice scores.

    Args:
        model_dir (str): path, where model and config files are inside. The directory should contain the files:
        model.pt, config.p, settings.json and lookuptable.json
        gpu (bool): 1 - run model with gpu
        smoothing_sigma (float): standard-deviation of gaussian filter for smoothing the slice-scores [in mm]
    """

    def __init__(
        self,
        model_dir: str,
        gpu: bool = 1, 
        smoothing_sigma: float = 10,
    ):  
        self.model = self._load_model(model_dir)
        # load settings
        with open(model_dir + "settings.json", "r") as f:
            settings = json.load(f)
            self.slope_mean = settings["slope_mean"]
            self.slope_std = settings["slope_std"]
            self.lower_bound_score = settings["lower_bound_score"]
            self.upper_bound_score = settings["upper_bound_score"]

        # load lookup table
        with open(model_dir + "lookuptable.json", "r") as f:
            table = json.load(f)
            # convert to integer keys
            self.lookup_table = {int(key): table[key] for key in table.keys()}

        self.smoothing_sigma = smoothing_sigma
        self.n2n = Nifti2Npy(
            target_pixel_spacing=3.5, min_hu=-1000, max_hu=1500, size=128
        )

    def _load_model(self, base_dir): 
	        self.base_filepath = base_dir
	        self.config_filepath = base_dir + "config.p"
	        self.model_filepath = base_dir + "model.pt"
	
	        with open(self.config_filepath, "rb") as f: 
	            self.config = pickle.load(f)
	            
	        self.model = BodyPartRegression(alpha=self.config["alpha"], 
	                                        lr=self.config["lr"], 
	                                        base_model=self.config["base_model"])
	        self.model.load_state_dict(torch.load(self.model_filepath))
	        self.model.eval()
	        self.model.to(self.device)

    def _body_range(self, slice_scores, lower_slice_index, upper_bound_score):
        body_region = np.where(slice_scores < upper_bound_score)[0]
        if len(body_region) == 0:
            return np.array([], dtype=int), 0
        upper_slice_index = np.max(body_region) + 1
        return np.arange(lower_slice_index, upper_slice_index), upper_slice_index

    def _get_body_part_examined(self, slice_scores):
        legs_array, legs_max = self._body_range(
            slice_scores, 0, self.lookup_table[0]["mean"]
        )
        pelvis_array, pelvis_max = self._body_range(
            slice_scores, legs_max, self.lookup_table[1]["mean"]
        )
        abdomen_array, abdomen_max = self._body_range(
            slice_scores, pelvis_max, self.lookup_table[3]["mean"]
        )
        lung_array, lung_max = self._body_range(
            slice_scores, abdomen_max, self.lookup_table[5]["mean"]
        )
        neck_array, neck_max = self._body_range(
            slice_scores, lung_max, self.lookup_table[6]["mean"]
        )
        head_array = np.arange(neck_max, len(slice_scores))

        legs_indices = list(legs_array.astype(float))
        pelvis_indices = list(pelvis_array.astype(float))
        abdomen_indices = list(abdomen_array.astype(float))
        lung_indices = list(lung_array.astype(float))
        neck_indices = list(neck_array.astype(float))
        head_indices = list(head_array.astype(float))

        body_part_examined = {
            "legs": legs_indices,
            "pelvis": pelvis_indices,
            "abdomen": abdomen_indices,
            "lungs": lung_indices,
            "shoulder-neck": neck_indices,
            "head": head_indices,
        }

        return body_part_examined

    def predict_scores(self, nifti_path):
        # get nifti file as tensor
        x, pixel_spacings = self.n2n.preprocess_nifti(nifti_path)
        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x)
        x_tensor.to(self.device)


        # predict slice-scores
        scores = self.predict_tensor(x_tensor)

        return scores, pixel_spacings

    def _transform_0_100(self, score, min_value=False):
        if isinstance(min_value, bool):
            min_value = self.lookup_table[0]["mean"]
        max_value = self.lookup_table[8]["mean"]

        score = score - min_value
        score = score * 100 / (max_value - min_value)

        return score

    def _transform_lookup(self):
        lookup_copy = {key: {} for key in self.lookup_table}
        for key in self.lookup_table:
            lookup_copy[key]["landmark-name"] = self.lookup_table[key]["landmark-name"]
            lookup_copy[key]["mean"] = np.round(
                self._transform_0_100(self.lookup_table[key]["mean"]), 3
            )
            lookup_copy[key]["std"] = np.round(
                self._transform_0_100(self.lookup_table[key]["std"], min_value=0), 3
            )

        return lookup_copy


    def predict(self, nifti_path, output_path):
        scores, pixel_spacings = self.predict_scores(nifti_path)

        # validation
        dsc = DataSanityCheck(
            scores,
            pixel_spacings[2],
            self.slope_mean,
            self.slope_std,
            self.lower_bound_score,
            self.upper_bound_score,
            self.smoothing_sigma,
        )
        cleaned_scores, valid_indices = dsc.remove_invalid_regions(scores)
        reverse_zordering = dsc.is_reverse_zordering()
        valid_zspacing = dsc.is_valid_zspacing()

        # estimate BodyPartExamined
        body_part_examined = self._get_body_part_examined(cleaned_scores)

        # transform to 0-100
        cleaned_scores = self._transform_0_100(cleaned_scores)
        scores = self._transform_0_100(scores)
        lookup_transformed = self._transform_lookup()

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
            "expected z-spacig": np.round(dsc.zhat, 2),
        }
        
        with open(output_path, "w") as f:
            json.dump(json_output, f)

        return json_output


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
