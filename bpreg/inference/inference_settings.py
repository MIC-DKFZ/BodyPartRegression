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

from bpreg.inference.inference_model import InferenceModel
from bpreg.dataset.bpr_dataset import BPRDataset
from bpreg.settings.settings import DATA_PATH, DF_DATA_SOURCE_PATH, LANDMARK_PATH
import numpy as np
import json

from tqdm import tqdm
from dataclasses import dataclass, field
from bpreg.score_processing import Scores
from bpreg.evaluation.evaluation import Evaluation
from torch.utils.data import Dataset


def postprocess_model_for_inference(
    model_path: str,
    df_data_source_path: str = DF_DATA_SOURCE_PATH,
    landmark_path: str = LANDMARK_PATH,
    data_path: str = DATA_PATH,
    upper_tangential_slope_quantile: float = 0.995,
    lower_tangential_slope_quantile: float = 0.005,
    transform_min_landmark: str = "pelvis_start",
    transform_max_landmark: str = "eyes_end",
    skip_slopes: bool = False,
):
    """Create inference-settings.json for model.
    The inference-settings.json file saves all relevant information which is needed for using the model during test time.

    Args:
        model_path (str): path of directory, where model.pt and config.json file is inside
        df_data_source_path (str, optional): path of the excel file which saves the train/val/test split
        landmark_path (str, optional): path of the excel file which saves the annotations for the train/val and test data
        upper_tangential_slope_quantile (float, optional): quantile of upper boundary for valid tangential slopes
        lower_tangential_slope_quantile (float, optional):  quantile of lower boundary for valid tangential slopes
        transform_min_landmark (str, optional): landmark which should get mapped to 0 after transformation
        transform_max_landmark (str, optional):  landmark which should get mapped to 100 after transformation
    """

    # get model evaluation
    print("Initialize model.")
    modelEval = Evaluation(
        model_path,
        df_data_source_path=df_data_source_path,
        landmark_path=landmark_path,
        data_path=data_path,
        landmark_start=transform_min_landmark,
        landmark_end=transform_max_landmark,
    )

    # get lookuptables
    print("Load lookuptables")
    lookuptable_train_original = modelEval.landmark_score_bundle.dict[
        "train"
    ].lookuptable
    lookuptable_train_transformed = modelEval.landmark_score_bundle.dict[
        "train"
    ].transformed_lookuptable

    lookuptable_train = {
        "original": lookuptable_train_original,
        "transformed": lookuptable_train_transformed,
    }

    lookuptable_train_val_original = modelEval.landmark_score_bundle.dict[
        "train+val-all-landmarks"
    ].lookuptable
    lookuptable_train_val_transformed = modelEval.landmark_score_bundle.dict[
        "train+val-all-landmarks"
    ].transformed_lookuptable

    lookuptable_train_val = {
        "original": lookuptable_train_val_original,
        "transformed": lookuptable_train_val_transformed,
    }

    # get min transform and max transform from train + val lookuptable
    transform_min = lookuptable_train_val_original[transform_min_landmark]["mean"]
    transform_max = lookuptable_train_val_original[transform_max_landmark]["mean"]

    settings = {
        "upper_tangential_quantile": upper_tangential_slope_quantile,
        "lower_tangential_quantile": lower_tangential_slope_quantile,
        "start-landmark": transform_min_landmark,
        "end-landmark": transform_max_landmark,
    }

    if not skip_slopes:
        # Tangential Slopes
        print("Compute tangential slopes of training data set")
        tangential_slopes = compute_tangential_slopes(
            modelEval.inference_model,
            modelEval.train_dataset,
            transform_min,
            transform_max,
        )
        upper_tangential_slope = np.nanquantile(
            tangential_slopes, upper_tangential_slope_quantile
        )
        lower_tangential_slope = np.nanquantile(
            tangential_slopes, lower_tangential_slope_quantile
        )
        tangential_slope_mean = np.nanmean(tangential_slopes)

        # Slice score curve slopes
        print("Compute slice score curve slopes")
        slice_score_curve_slopes = compute_slice_score_curve_slopes(
            modelEval.inference_model,
            modelEval.train_dataset,
            lower_tangential_slope,
            upper_tangential_slope,
            transform_min,
            transform_max,
        )

        slope_mean = np.nanmean(slice_score_curve_slopes)
        slope_std = np.nanstd(slice_score_curve_slopes, ddof=1)
        slope_median = np.nanquantile(slice_score_curve_slopes, 0.5)

        storage = InferenceSettingsStorage(
            slope_mean=slope_mean,
            slope_median=slope_median,
            slope_std=slope_std,
            upper_quantile_tangential_slope=upper_tangential_slope,
            lower_quantile_tangential_slope=lower_tangential_slope,
            tangential_slope_mean=tangential_slope_mean,
            lookuptable_train=lookuptable_train,
            lookuptable_train_val=lookuptable_train_val,
            settings=settings,
        )

    else:
        storage = InferenceSettingsStorage(
            lookuptable_train=lookuptable_train,
            lookuptable_train_val=lookuptable_train_val,
            settings=settings,
        )

    storage.save(model_path)


def compute_tangential_slopes(
    inference_model: InferenceModel,
    train_dataset: BPRDataset,
    transform_min: float,
    transform_max: float,
):
    tangential_slopes = []
    for i in tqdm(range(len(train_dataset))):
        z = train_dataset.z_spacings[i]
        X = train_dataset.get_full_volume(i)
        scores = inference_model.predict_npy_array(X)
        scores = Scores(
            scores,
            z,
            transform_min=transform_min,
            transform_max=transform_max,
            tangential_slope_min=-np.inf,
            tangential_slope_max=np.inf,
        )
        tangential_slopes += list(scores.slopes)

    return np.array(tangential_slopes)


def compute_slice_score_curve_slopes(
    inference_model: InferenceModel,
    train_dataset: BPRDataset,
    lower_tangential_slope: float,
    upper_tangential_slope: float,
    transform_min: float,
    transform_max: float,
):
    curve_slopes = []
    for i in tqdm(range(len(train_dataset))):
        z = train_dataset.z_spacings[i]
        X = train_dataset.get_full_volume(i)
        scores = inference_model.predict_npy_array(X)
        scores = Scores(
            scores,
            z,
            transform_min=transform_min,
            transform_max=transform_max,
            tangential_slope_min=lower_tangential_slope,
            tangential_slope_max=upper_tangential_slope,
        )
        curve_slopes.append(scores.a)

    return np.array(curve_slopes)


@dataclass
class InferenceSettingsStorage:
    slope_mean: float = 0.118
    slope_median: float = 0.118
    slope_std: float = 0.012
    tangential_slope_mean: float = 0.113
    upper_quantile_tangential_slope: float = 0.25
    lower_quantile_tangential_slope: float = -0.037
    lookuptable_train: dict = field(default_factory={})
    lookuptable_train_val: dict = field(default_factory={})
    settings: dict = field(default_factory={})

    def __post_init__(self):
        self.json_dict = self.__dict__.copy()

    def save(self, save_path=""):
        with open(save_path + "inference-settings.json", "w") as f:
            json.dump(self.json_dict, f, indent=4)

    def load(self, path=""):
        with open(path, "r") as f:
            json_dict = json.load(f)

        self.__init__(**json_dict)
        self.__post_init__()
