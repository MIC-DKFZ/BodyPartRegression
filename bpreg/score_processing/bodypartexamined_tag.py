import numpy as np
from collections import Counter

from bpreg.score_processing.scores import Scores
from bpreg.score_processing.bodypartexamined_dict import get_scoreDict

BODY_PARTS_INCLUDED = {
    "PELVIS": ["pelvis_start", "femur_end", "pelvis_end"],
    "ABDOMEN": [
        "L5",
        "L4",
        "L3",
        "L2",
        "L1",
        "Th12",
        "Th11",
        "Th10",
        "Th9",
    ],
    "CHEST": [
        "Th12",
        "Th11",
        "Th10",
        "Th9",
        "Th8",
        "Th7",
        "Th6",
        "Th5",
        "Th4",
        "Th3",
        "Th2",
        "Th1",
    ],
    "NECK": ["Th3", "Th2", "Th1", "C7", "C6", "C5", "C4", "C3", "C2"],
    "HEAD": ["C3", "C2", "C1", "eyes_end"],
}

MIN_PRESENT_LANDMARKS = {"PELVIS": 2, "ABDOMEN": 7, "CHEST": 9, "NECK": 6, "HEAD": 3}

DISTINCT_BODY_PARTS = {
    "PELVIS": ["pelvis_start", "pelvis_end"],
    "ABDOMEN": ["pelvis_end", "Th9"],
    "CHEST": ["Th9", "Th2"],
    "NECK": ["Th2", "C3"],
    "HEAD": ["C3", "head_end"],
}


class BodyPartExaminedTag:
    def __init__(
        self,
        lookuptable,
        body_parts_included=BODY_PARTS_INCLUDED,
        distinct_body_parts=DISTINCT_BODY_PARTS,
        min_present_landmarks=MIN_PRESENT_LANDMARKS,
        zrange_threshold=100,  # in mm
        ignore_invalid_z: bool = False,
    ):
        self.ignore_invalid_z = ignore_invalid_z
        self.body_parts_included = body_parts_included
        self.min_present_landmarks = min_present_landmarks
        self.distinct_body_parts = distinct_body_parts
        self.lookuptable = lookuptable
        self.zrange_threshold = zrange_threshold

        self.body_parts_included_scores = get_scoreDict(
            self.body_parts_included, lookuptable
        )
        self.distinct_body_parts_scores = get_scoreDict(
            self.distinct_body_parts, lookuptable
        )

    def estimate_tag(self, scores: Scores):
        # if z-spacing is invalid return NONE as body part examined
        if (scores.valid_zspacing == 0) and not self.ignore_invalid_z:
            return "NONE"

        # check if z-range is greater than 100 mm
        zrange = len(scores.original_values) * scores.zspacing

        tag = self.get_bodypartexamined_from_volume(scores.values)
        if zrange < self.zrange_threshold or (isinstance(tag, float) and np.isnan(tag)):
            tag = self.get_most_frequent_bodypartexamined_in_slices(scores.values)

        return tag

    def is_landmark_present(self, scores: np.array, landmark_score: float) -> bool:
        smaller_scores = np.where(scores < landmark_score)[0]
        bigger_scores = np.where(scores > landmark_score)[0]

        if (len(smaller_scores) > 0) and (len(bigger_scores) > 0):
            return True
        return False

    def is_bodypart_present(
        self, scores: np.array, landmark_scores, landmark_threshold
    ) -> bool:
        landmarks_inside = 0
        for landmark_score in landmark_scores:
            if self.is_landmark_present(scores, landmark_score):
                landmarks_inside += 1

        if landmarks_inside >= landmark_threshold:
            return True
        return False

    def is_score_in_body_part(self, score, bodypart) -> bool:
        bodypart_boundaries = self.distinct_body_parts_scores[bodypart]
        if (score > bodypart_boundaries[0]) & (score < bodypart_boundaries[1]):
            return True
        return False

    def get_most_frequent_bodypartexamined_in_slices(self, scores: np.array) -> str:
        bodyparts_for_slices = []
        for score in scores:
            for bodypart in self.distinct_body_parts_scores:
                if self.is_score_in_body_part(score, bodypart):
                    bodyparts_for_slices.append(bodypart)
                    continue

        if len(bodyparts_for_slices) == 0:
            return "NONE"

        return most_frequent(bodyparts_for_slices)

    def get_bodypartexamined_from_volume(self, scores: np.array) -> str:
        bodyparts_included = []
        for bodypart, landmark_scores in self.body_parts_included_scores.items():
            threshold = self.min_present_landmarks[bodypart]
            if self.is_bodypart_present(scores, landmark_scores, threshold):
                bodyparts_included.append(bodypart)

        return self.join_bodyparts_included(bodyparts_included)

    def join_bodyparts_included(self, bodyparts_list: list):
        if len(bodyparts_list) == 0:
            return np.nan

        if len(bodyparts_list) == 1:
            return bodyparts_list[0]

        if "HEAD" in bodyparts_list:
            if "PELVIS" in bodyparts_list:
                return "HEAD-NECK-CHEST-ABDOMEN-PELVIS"
            elif "ABDOMEN" in bodyparts_list:
                return "HEAD-NECK-CHEST-ABDOMEN"
            elif "CHEST" in bodyparts_list:
                return "HEAD-NECK-CHEST"
            else:
                return "HEAD-NECK"

        if "NECK" in bodyparts_list:
            if "PELVIS" in bodyparts_list:
                return "NECK-CHEST-ABDOMEN-PELVIS"
            elif "ABDOMEN" in bodyparts_list:
                return "NECK-CHEST-ABDOMEN"
            else:
                return "NECK-CHEST"

        if "CHEST" in bodyparts_list:
            if "PELVIS" in bodyparts_list:
                return "CHEST-ABDOMEN-PELVIS"
            else:
                return "CHEST-ABDOMEN"

        return "ABDOMEN-PELVIS"


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
