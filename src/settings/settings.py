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
    10: CLASSES["head"],
    11: CLASSES["head"],
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
