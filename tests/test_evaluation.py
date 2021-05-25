import numpy as np 
import sys
sys.path.append("../")
from scripts.network_architecture.bpr_model import normalized_mse_from_matrices


def test_normalized_mse(): 
    # test normalized mse with full matrices
    tm = np.array([[1, 2, 4], [1.5, 2.2, 3.5], [1, 1.9, 4.1]])
    vm = np.array([[1.1, 2.1, 3.9], [0.8, 2.0, 4.2]])

    mse, mse_std, d = normalized_mse_from_matrices(vm, tm)

    assert d == 2.7
    assert np.round(mse, 5) == 0.00587
    assert np.round(mse_std, 5) == 0.00319

    # test normalized mse with nan-values
    tm = np.array([[1, 2, 4], [np.nan, 2.2, 3.5], [1, 1.9, 4.1]])
    vm = np.array([[1.1, 2.1, np.nan], [0.8, 2.0, 4.2]])

    mse, mse_std, d = normalized_mse_from_matrices(vm, tm)

    assert np.round(d, 2) == 2.87 
    assert np.round(mse, 4) == 0.0041
    assert np.round(mse_std, 4) == 0.0022
