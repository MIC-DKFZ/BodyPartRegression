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

import argparse
from bpreg.inference.inference_settings import postprocess_model_for_inference
from bpreg.settings import *


def main():
    """
    create inference-settings.json file for model to use model for inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DF_DATA_SOURCE_PATH)
    parser.add_argument("--landmarks", default=LANDMARK_PATH)
    parser.add_argument("--uq", default=0.995)
    parser.add_argument("--lq", default=0.005)
    parser.add_argument("--minl", default="pelvis_start")
    parser.add_argument("--maxl", default="eyes_end")

    value = parser.parse_args()
    model_path = value.model
    df_data_source_path = value.data
    landmark_path = value.landmarks
    upper_quantile = value.uq
    lower_quantile = value.lq
    min_landmark = value.minl
    max_landmark = value.maxl

    postprocess_model_for_inference(
        model_path,
        df_data_source_path,
        landmark_path,
        upper_tangential_slope_quantile=upper_quantile,
        lower_tangential_slope_quantile=lower_quantile,
        transform_min_landmark=min_landmark,
        transform_max_landmark=max_landmark,
    )
