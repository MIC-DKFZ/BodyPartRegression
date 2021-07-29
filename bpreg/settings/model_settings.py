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
import sys, json, pickle

sys.path.append("../../")

from dataclasses import dataclass, field
from torchvision import transforms
import albumentations as A

from bpreg.settings.settings import *
from bpreg.dataset.custom_transformations import *


@dataclass
class ModelSettings:
    """Create settings file for training a body part regression model."""

    df_data_source_path: str = ""
    data_path: str = ""
    landmark_path: str = ""
    save_dir: str = ""
    batch_size: int = 64
    effective_batch_size: int = 64
    equidistance_range: list = field(default_factory=lambda: [5, 100])
    num_slices: int = 4
    epochs: int = 480
    alpha_h: float = 1
    beta_h: float = 0.01
    loss_order: str = "h"
    lambda_: float = 0
    alpha: float = 0
    lr: float = 1e-4
    shuffle_train_dataloader: bool = True
    random_seed: int = 0
    deterministic: bool = True
    save_model: bool = True
    base_model: str = "vgg"
    transform_params: dict = field(default_factory=lambda: TRANSFORM_STANDARD_PARAMS)
    name: str = "default.p"
    model_name: str = "standard"
    model: str = ""

    def __post_init__(self):
        self.json_dict = self.__dict__.copy()

        self.filepath = self.save_dir + self.name
        self.custom_transform = self.get_custom_transform()
        self.albumentation_transform = self.get_albumentation_transform()

    def get_custom_transform(self):
        custom_transforms = []
        for transform in [GaussNoise, ShiftHU, ScaleHU, AddFrame]:
            transform_name = transform.__name__
            if not transform_name in self.transform_params:
                continue
            params = self.transform_params[transform_name]
            custom_transforms.append(transform(**params))
        return transforms.Compose(custom_transforms)

    def get_albumentation_transform(self):
        albumetnation_transforms = []
        for transform in [A.Flip, A.Transpose, A.ShiftScaleRotate, A.GaussianBlur]:
            transform_name = transform.__name__
            if not transform_name in self.transform_params:
                continue
            params = self.transform_params[transform_name]
            albumetnation_transforms.append(transform(**params))
        return A.Compose(albumetnation_transforms)

    def save(self, save_path=""):
        for key in [
            "albumentation_transform",
            "custom_transform",
            "filepath",
            "json_dict",
        ]:
            if key in self.json_dict:
                del self.json_dict[key]

        if len(save_path) == "":
            save_path = self.filepath
        with open(save_path, "w") as f:
            json.dump(self.json_dict, f, indent=4)

    def load(self, path=""):
        with open(path, "r") as f:
            json_dict = json.load(f)

        self.__init__(**json_dict)
        self.__post_init__()

    def load_pickle(self, path):
        """Provide backward compatibility with outdated pickle files."""

        delete_keys = [
            "custom_transform",
            "albumentation_transform",
            "custom_transform_params",
            "albumentation_transform_params",
            " ",
            "   ",
            "     ",
            "    ",
            "description",
            "accumulate_grad_batches",
            "lambda",
            "test_loss",
            "test_loss_order",
            "test_loss_dist",
            "test_loss_l2",
            "test_landmark_metric_mean",
            "test_landmark_metric_var",
            "pearson-correlation",
            "validation loss",
            "landmark metric",
            "pre-name",
        ]

        with open(path, "rb") as f:
            config = pickle.load(f)

        custom_transform = config["custom_transform"]
        albumentation_transform = config["albumentation_transform"]
        config["lambda_"] = config["lambda"]

        # delete keys which this class does not need
        for key in delete_keys:
            if key in config.keys():
                del config[key]

        # overwrite parameters
        self.__init__(**config)
        self.custom_transform = custom_transform
        self.albumentation_transform = albumentation_transform

        # get transform parameters
        params = self.transforms_to_dict(
            self.custom_transform.__dict__["transforms"],
            ["square_frame", "circle_frame"],
        )
        params_albumentation = self.transforms_to_dict(
            self.albumentation_transform,
            [
                "deterministic",
                "save_key",
                "replay_mode",
                "mask_value",
                "applied_in_replay",
                "params",
            ],
            albumentation=True,
        )
        params.update(params_albumentation)

        # if "ShiftHU" in params.keys():
        #   params["ShiftHU"]["limit"] = params["ShiftHU"]["shift_limit"]
        #   del params["ShiftHU"]["shift_limit"]

        if "AddFrame" in params.keys():
            params["AddFrame"]["dimension"] = params["AddFrame"]["d"]
            del params["AddFrame"]["d"]

        self.transform_params = params
        self.json_dict = self.__dict__.copy()

    def transforms_to_dict(self, transform_list, delete_keys, albumentation=False):
        params = {}
        for transform in transform_list:
            name = type(transform).__name__

            transform_params = transform.__dict__
            if albumentation:
                transform_params = transform.get_base_init_args()
                transform_params.update(transform.get_transform_init_args())

            params[name] = transform_params

            # filter private keys
            params[name] = {
                key: params[name][key]
                for key in params[name].keys()
                if not key.startswith("_")
            }

            for key in delete_keys:
                if key in params[name].keys():
                    del params[name][key]

        return params

    def __str__(self):
        start_line = "\nBODY PART REGRESSION MODEL SETTINGS\n"
        line = "*******************************************************\n"
        part1_keys = [
            "model_name",
            "name",
            "df_data_source_path",
            "data_path",
            "landmark_path",
            "save_dir",
            "shuffle_train_dataloader",
            "random_seed",
            "deterministic",
            "save_model",
            "base_model",
        ]
        part2_keys = [
            "batch_size",
            "effective_batch_size",
            "equidistance_range",
            "num_slices",
        ]
        part3_keys = [
            "alpha_h",
            "beta_h",
            "loss_order",
            "lambda_",
            "alpha",
            "lr",
            "epochs",
        ]

        my_string = start_line + line
        for part in [part1_keys, part2_keys, part3_keys]:
            for key in part:
                my_string += f"{key:<28}:\t{self.json_dict[key]}\n"
            my_string += line
        my_string += "\n"

        return my_string
