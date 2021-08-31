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

import os 
from os import getenv
from os.path import join, exists
from glob import glob
from pathlib import Path

from bpreg.inference import InferenceModel


gpu_available=0
stringify_json=1
model_base_dir = "src/models/public_bpr_model/"
model_inference = InferenceModel(model_base_dir, gpu=gpu_available)

element_input_dir = "data/test_cases/"
element_output_dir = "data/test_results/"

for filename in os.listdir(element_input_dir): 
    json_filename = filename.replace(".nii", "").replace(".gz", "") + ".json"
    input_path = join(element_input_dir, filename)
    output_path = join(element_output_dir, json_filename)
    
    print(f"Save .json file: {output_path}")
    model_inference.nifti2json(nifti_path=input_path, 
                              output_path=output_path, 
                              stringify_json=stringify_json)

