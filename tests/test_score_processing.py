import numpy as np 
import sys, os

sys.path.append("../")
from bpreg.score_processing.scores import Scores

scores1 = np.array([2, -2, -1, 0, 2, 4, 7, 10, 13, 16, 20], dtype=float)
scores2 = np.array([50, 61, 72, 83, 90, 105, 95, 90, 85], dtype=float)
scores3 = np.array([5, 2, -10, 0, 10, 20, 30, 40, 50, 60, 80, 90, 100, 110, -10, -10 ], dtype=float)
scores4 = np.array([110, 100, -10, 0, 10, 20, 30, 40, 50, 60, 80, 90, 100, 110, -10, -10 ], dtype=float)


# TODO - fix tests!!! 

"""
def test_valid_range(): 
    scores = Scores(scores1, zspacing=20, smoothing_sigma=1)
    np.testing.assert_equal(np.where(np.isnan(scores.values))[0], np.array([0]))

    scores = Scores(scores2, zspacing=40, smoothing_sigma=1)
    np.testing.assert_equal(np.where(np.isnan(scores.values))[0], np.array([6, 7, 8]))

    scores = Scores(scores3, zspacing=80, smoothing_sigma=0)
    np.testing.assert_equal(np.where(np.isnan(scores.values))[0], np.array([0, 1, 14, 15]))

    scores = Scores(scores4, zspacing=80, smoothing_sigma=0)
    np.testing.assert_equal(np.where(np.isnan(scores.values))[0], np.array([0, 1, 14, 15]))
"""

def test_slope(): 
    pass

def test_transform(): 
    pass

def test_cut_extreme_diffs(): 
    pass

if __name__ == "__main__": 
    test_valid_range()