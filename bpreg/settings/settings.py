import numpy as np
import cv2, os, json

cv2.setNumThreads(1)

abspath = os.path.abspath(__file__)
MAIN_PATH = os.path.dirname(os.path.dirname(os.path.dirname(abspath)))

# Path for default inference model
DEFAULT_MODEL = os.path.join(MAIN_PATH, "src/models/public_bpr_model/")

# define landmark names
LANDMARK_NAMES = [
    "pelvis-start",
    "femur-end",
    "L5",
    "L3",
    "L1",
    "Th11",
    "Th8",
    "Th5",
    "Th2",
    "C6",
    "C1",
    "eyes-end",
]

# define classes
# 0 - legs, 1 - pelvis, 2 - abdomen
CLASSES = {"legs": 0, "pelvis": 1, "abdomen": 2, "thorax": 3, "neck": 4, "head": 5}

LANDMARK_CLASS_MAPPING = {
    0: CLASSES["legs"],
    1: CLASSES["pelvis"],
    2: CLASSES["pelvis"],
    3: CLASSES["abdomen"],
    4: CLASSES["abdomen"],
    5: CLASSES["abdomen"],
    6: CLASSES["thorax"],
    7: CLASSES["thorax"],
    8: CLASSES["thorax"],
    9: CLASSES["neck"],
    10: CLASSES["neck"],
    11: CLASSES["head"],
}

CLASS_TO_LANDMARK_5 = {
    CLASSES["pelvis"]: [0, 2],
    CLASSES["abdomen"]: [2, 5],
    CLASSES["thorax"]: [5, 8],
    CLASSES["neck"]: [8, 10],
    CLASSES["head"]: [10, 11],
}

CLASS_TO_LANDMARK_3 = {
    CLASSES["pelvis"]: [0, 2],
    CLASSES["abdomen"]: [2, 5],
    CLASSES["thorax"]: [5, 8],
}

# define colors for plots
COLORS = [
    "black",
    "gray",
    "silver",
    "lightcoral",
    "brown",
    "chocolate",
    "goldenrod",
    "red",
    "lightgreen",
    "green",
    "deepskyblue",
    "steelblue",
    "mediumorchid",
    "plum",
    "purple",
    "pink",
]

DF_DATA_SOURCE_PATH = "/home/AD/s429r/Documents/Data/DataSet/MetaData/meta-data-public-dataset-npy-arrays-3.5mm-windowing-sigma.xlsx"
LANDMARK_PATH = (
    "/home/AD/s429r/Documents/Data/DataSet/MetaData/landmarks-meta-data-v2.xlsx"
)
DATA_PATH = "/home/AD/s429r/Documents/Data/DataSet/Arrays-3.5mm-sigma-01/"

BODY_PARTS = {
    "legs": [np.nan, "pelvis_start"],
    "pelvis": ["pelvis_start", "pelvis_end"],
    "abdomen": ["L5", "Th8"],
    "chest": ["Th12", "Th1"],
    "shoulder-neck": ["Th3", "C2"],
    "head": ["C5", np.nan],
}

TRANSFORM_STANDARD_PARAMS = {
    "GaussNoise": {
        "std_min": 0,
        "std_max": 0.04,
        "min_value": -1,
        "max_value": 1,
        "p": 0.5,
    },
    "ShiftHU": {"limit": 0.08, "p": 0.5, "min_value": -1, "max_value": 1},
    "ScaleHU": {"p": 0.5, "scale_delta": 0.2, "max_value": 1, "min_value": -1},
    "AddFrame": {"p": 0.25, "dimension": 128, "r_circle": 0.75, "fill_value": -1},
    "Flip": {"p": 0.5},
    "Transpose": {"p": 0.5},
    "ShiftScaleRotate": {
        "shift_limit": 0,
        "scale_limit": 0.2,
        "rotate_limit": 10,
        "p": 0.5,
        "border_mode": cv2.BORDER_REFLECT_101,
    },
    "GaussianBlur": {
        "blur_limit": (3, 7),
        "sigma_limit": 0.5,
        "always_apply": False,
        "p": 0.5,
    },
}
