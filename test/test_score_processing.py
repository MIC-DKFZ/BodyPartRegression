import numpy as np
from bpreg.score_processing import Scores


def create_scores(slope, error, n, zspacing, b=0):
    epsilon = np.random.normal(0, error, n)
    x = np.arange(0, n, 1)
    y = x * zspacing * slope + epsilon + b

    return y


def test_basic_score_processing():
    slope = 0.11
    zspacing = 1
    y = create_scores(slope=slope, error=0.2, n=100, zspacing=zspacing)

    scores = Scores(
        y, zspacing=zspacing, transform_min=0, transform_max=100, slope_mean=slope
    )

    # Check if no missing values are in cleaned slice scores
    assert np.sum(np.isnan(scores.values) * 1) == 0

    # check if zspacing is valid
    assert scores.valid_zspacing == 1


def test_tail_missing_values():
    zspacing = 4
    slope = 0.11
    y = create_scores(slope=0.11, error=0.2, n=400, zspacing=zspacing, b=-20)
    y = [20, 20, 20] + list(y) + [20, 20, 20]

    scores = Scores(
        y, zspacing=zspacing, transform_min=0, transform_max=100, slope_mean=slope
    )
    assert np.sum(np.isnan(scores.values[0:3])) == 3
    assert np.sum(np.isnan(scores.values[-3:])) == 3


def test_transform():
    y = np.arange(0, 10)
    scores = Scores(y, zspacing=1, transform_min=1, transform_max=8, slope_mean=1)
    assert scores.original_transformed_values[1] == 0
    assert scores.original_transformed_values[-2] == 100


def test_valid_zspacing():
    slope = 0.11
    zspacing = 1
    y = create_scores(slope=slope, error=0.2, n=100, zspacing=zspacing)

    scores = Scores(
        y, zspacing=zspacing, transform_min=0, transform_max=100, slope_mean=2 * slope
    )
    assert scores.valid_zspacing == 0

    scores = Scores(
        y, zspacing=zspacing, transform_min=0, transform_max=100, slope_mean=0.5 * slope
    )
    assert scores.valid_zspacing == 0
