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

import sys, os, argparse

sys.path.append("../")
from bpreg.settings import ModelSettings

data_path = {
    "local": "/home/AD/s429r/Documents/Data/DataSet/",
    "cluster": "/gpu/data/OE0441/s429r/",
}

save_path = {
    "local": "/home/AD/s429r/Documents/Data/Results/body-part-regression-models/",
    "cluster": "/gpu/checkpoints/OE0441/s429r/results/bodypartregression/",
}


def base_config(mode: str = "cluster"):
    params = {
        "df_data_source_path": data_path[mode]
        + "MetaData/meta-data-public-dataset-npy-arrays-3.5mm-windowing-sigma.xlsx",
        "data_path": data_path[mode] + "Arrays-3.5mm-sigma-01/",
        "landmark_path": data_path[mode] + "MetaData/landmarks-meta-data-v2.xlsx",
        "save_dir": save_path[mode],
    }
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="cluster")

    value = parser.parse_args()
    mode = value.mode

    experiments = {0: {"model_name": "standard", "name": "standard-2.p"}}
    base_path = "../src/configs/" + mode + "/"

    for idx, data in experiments.items():
        params = base_config(mode=mode)
        params.update(data)
        config = ModelSettings(**params)
        print(config)
        save_path_dir = base_path + config.model_name + "/"
        if not os.path.exists(save_path_dir):
            os.mkdir(save_path_dir)
        print(f"SAVE CONFIG: {save_path_dir + config.name}")
        config.save(save_path=save_path_dir + config.name)
