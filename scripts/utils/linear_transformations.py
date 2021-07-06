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

import numpy as np


def linear_transform(x, scale=1, min_value=0, max_value=1):
    x = x - min_value
    x = x * scale / (max_value - min_value)

    return x


def transform_0to100(
    score,
    lookuptable,
    min_value=np.nan,
    landmark_start="pelvis_start",
    landmark_end="eyes_end",
):
    if np.isnan(min_value):
        min_value = lookuptable[landmark_start]["mean"]
    max_value = lookuptable[landmark_end]["mean"]

    score = linear_transform(score, scale=100, min_value=min_value, max_value=max_value)
    return score


def transform_lookuptable(
    lookuptable, landmark_start="pelvis_start", landmark_end="eyes_end"
):
    lookup_copy = {key: {} for key in lookuptable}
    for key in lookuptable:
        lookup_copy[key]["mean"] = np.round(
            transform_0to100(
                lookuptable[key]["mean"],
                lookuptable,
                landmark_start=landmark_start,
                landmark_end=landmark_end,
            ),
            3,
        )
        lookup_copy[key]["std"] = np.round(
            transform_0to100(
                lookuptable[key]["std"],
                lookuptable,
                min_value=0.0,
                landmark_start=landmark_start,
                landmark_end=landmark_end,
            ),
            3,
        )

    return lookup_copy
