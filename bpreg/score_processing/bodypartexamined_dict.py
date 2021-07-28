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

import sys, os
import numpy as np

sys.path.append("../../")
from bpreg.settings.settings import *


class BodyPartExaminedDict:
    def __init__(self, lookuptable: dict, body_parts=BODY_PARTS):
        self.lookuptable = lookuptable
        self.landmarkDict = body_parts
        self.scoreDict = get_scoreDict(self.landmarkDict, self.lookuptable)

    def get_score_indices(self, scores, min_score=np.nan, max_score=np.nan):
        scores = np.array(scores)
        if ~np.isnan(min_score) & ~np.isnan(max_score):
            return np.where((scores >= min_score) & (scores < max_score))
        elif np.isnan(min_score) & ~np.isnan(max_score):
            return np.where(scores < max_score)
        elif ~np.isnan(min_score) & np.isnan(max_score):
            return np.where(scores > min_score)
        else:
            return np.arange(0, len(scores))

    def get_examined_body_part(self, scores):
        bodyPartDict = {}
        for bodypart, boundary_scores in self.scoreDict.items():
            indices = self.get_score_indices(
                scores, min_score=boundary_scores[0], max_score=boundary_scores[1]
            )[0]
            bodyPartDict[bodypart] = list(indices.astype(np.float64))

        return bodyPartDict


def get_scoreDict(landmarkDict: dict, lookuptable: dict) -> dict:
    scoreDict = {}
    for key, items in landmarkDict.items():
        scores = []
        for landmark in items:
            if isinstance(landmark, float):
                scores.append(landmark)
            else:
                scores.append(lookuptable[landmark]["mean"])

        scoreDict[key] = scores
    return scoreDict
