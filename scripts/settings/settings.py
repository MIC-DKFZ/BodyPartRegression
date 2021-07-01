import numpy as np 

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
    "eyes-end"
]

# define classes
# 0 - legs, 1 - pelvis, 2 - abdomen
CLASSES = {"legs": 0, "pelvis": 1, "abdomen": 2, "thorax": 3, "neck": 4, "head": 5}

LANDMARK_CLASS_MAPPING = { # TODO ######
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
    11: CLASSES["head"]}

CLASS_TO_LANDMARK_5 = {CLASSES["pelvis"]: [0, 2], 
                             CLASSES["abdomen"]: [2, 5], 
                             CLASSES["thorax"]: [5, 8], 
                             CLASSES["neck"]: [8, 10], 
                             CLASSES["head"]: [10, 11]}

CLASS_TO_LANDMARK_3 = {CLASSES["pelvis"]: [0, 2], 
                             CLASSES["abdomen"]: [2, 5], 
                             CLASSES["thorax"]: [5, 8]}

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
LANDMARK_PATH =  "/home/AD/s429r/Documents/Data/DataSet/MetaData/landmarks-meta-data-v2.xlsx"
DATA_PATH = "/home/AD/s429r/Documents/Data/DataSet/Arrays-3.5mm-sigma-01/"

BODY_PARTS = {
    "legs": [np.nan, "pelvis_start"], 
    "pelvis": ["pelvis_start", "pelvis_end"], 
    "abdomen": ["pelvis_end", "Th12"], 
    "chest": ["Th12", "Th2"], 
    "shoulder-neck": ["Th2", "C4"], 
    "head": ["C4", np.nan]
}