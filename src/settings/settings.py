# define landmark names
LANDMARK_NAMES = [
    "pelvis-start",
    "pelvis-end",
    "kidney",
    "lung-start",
    "liver-end",
    "lung-end",
    "teeth",
    "nose",
    "eyes-end",
]

# define classes
# 0 - legs, 1 - pelvis, 2 - abdomen
CLASSES = {"legs": 0, "pelvis": 1, "abdomen": 2, "thorax": 3, "neck": 4, "head": 5}

LANDMARK_CLASS_MAPPING = {
    0: CLASSES["legs"],
    1: CLASSES["pelvis"],
    2: CLASSES["abdomen"],
    3: CLASSES["abdomen"],
    4: CLASSES["abdomen"],
    5: CLASSES["thorax"],
    6: CLASSES["neck"],
    7: CLASSES["head"],
    8: CLASSES["head"],
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
