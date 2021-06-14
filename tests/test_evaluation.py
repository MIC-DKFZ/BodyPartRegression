import numpy as np 
import sys
sys.path.append("../")
from scripts.evaluation.accuracy import Accuracy
from scripts.evaluation.normalized_mse import NormalizedMSE

def test_normalized_mse(): 
    # test normalized mse with full matrices
    tm = np.array([[1, 2, 4], [1.5, 2.2, 3.5], [1, 1.9, 4.1]])
    vm = np.array([[1.1, 2.1, 3.9], [0.8, 2.0, 4.2]])

    normMSE = NormalizedMSE()
    mse, mse_std  = normMSE.from_matrices(vm, tm)

    assert d == 2.7
    assert np.round(mse, 5) == 0.00587
    assert np.round(mse_std, 5) == 0.00319

    # test normalized mse with nan-values
    tm = np.array([[1, 2, 4], [np.nan, 2.2, 3.5], [1, 1.9, 4.1]])
    vm = np.array([[1.1, 2.1, np.nan], [0.8, 2.0, 4.2]])

    mse, mse_std = normMSE.from_matrices(vm, tm)

    assert np.round(d, 2) == 2.87 
    assert np.round(mse, 4) == 0.0041
    assert np.round(mse_std, 4) == 0.0022

def test_accuracy_class_initalization(): 
    estimated_landmark_slice_scores = np.array([-10, -7.5, -6, -5.5, -4, -3.5, -2, 0, 2, 4.5, 6, 8])
    class_to_landmark = {0: [0, 2], 
                               1: [2, 5], 
                               2: [5, 8], 
                               3: [8, 10], 
                               4: [10, 11]}
    
    acc = Accuracy(estimated_landmark_slice_scores, class_to_landmark)
    
    # test class parameter
    assert acc.class_to_score_mapping == {0: [-10.0, -6.0],
                                         1: [-6.0, -3.5],
                                         2: [-3.5, 2.0],
                                         3: [2.0, 6.0],
                                         4: [6.0, 8.0]}
    
    estimated_landmark_slice_scores = np.array([-5, -2, 0, 1, 4])
    class_to_landmark = {0: [0, 1], 
                               1: [1, 3], 
                               2: [3, 4]}

    acc = Accuracy(estimated_landmark_slice_scores, class_to_landmark)
    assert acc.class_to_score_mapping == {0: [-5, -2], 1: [-2, 1], 2: [1, 4]}
    

    # test class prediction 

    
def test_class_prediction(): 
    estimated_landmark_slice_scores = np.array([-10, -7.5, -6, -5.5, -4, -3.5, -2, 0, 2, 4.5, 6, 8])
    class_to_landmark = {0: [0, 2], 
                               1: [2, 5], 
                               2: [5, 8], 
                               3: [8, 10], 
                               4: [10, 11]}
    
    slice_scores = np.array([-4. , -3.5, -3. , -2.5, -2. , 
                          -1.5, -1. , -0.5,  0. ,  0.5,  
                          1. ,  1.5,  2. ,  2.5])
    predicted_classes = np.array([1., 2., 2., 2., 2., 2., 2., 2., 2., 2.,  2., 2., 3., 3.])
    acc = Accuracy(estimated_landmark_slice_scores, class_to_landmark)
    np.testing.assert_equal(acc.class_prediction(slice_scores), predicted_classes) 
        
    
    slice_scores = np.arange(5.5, 9, 0.5)
    predicted_classes = np.array([ 3.,  4.,  4.,  4.,  4., np.nan, np.nan])
    np.testing.assert_equal(acc.class_prediction(slice_scores), predicted_classes) 
    
def test_ground_truth_class(): 
    estimated_landmark_slice_scores = np.array([-5, -2, 0, 1, 4])
    landmark_positions = np.array([np.nan, np.nan, np.nan, 5, 8])
    class_to_landmark = {0: [0, 1], 
                         1: [1, 3], 
                         2: [3, 4]}

    acc = Accuracy(estimated_landmark_slice_scores, class_to_landmark)
    ground_truth_classes = acc.ground_truth_class(landmark_positions, max_slices=10)
    np.testing.assert_equal(ground_truth_classes, 
                            np.array([ np.nan, np.nan, np.nan, np.nan, np.nan,  2.,  2.,  2., np.nan, np.nan])) 

def test_volume(): 
    class_to_landmark = {0: [0, 1], 
                     1: [1, 3], 
                     2: [3, 4]}
    expected_scores = np.array([-5, -2.5, 0, 1, 3])
    acc = Accuracy(expected_scores, class_to_landmark)

    slice_scores = np.array([-7, -6.5, -6, -5, -4, -3, -2, -1, -0.5, 0, 1, 1.2, 1.5, 2])
    landmarks = np.array([3, 5, 6, 8, np.nan])
    class_prediction = np.array([np.nan, np.nan, np.nan, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    groundtruth_prediction = np.array([np.nan, np.nan, np.nan, 0, 0, 1, 1, 1, 
                                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_equal(acc.class_prediction(slice_scores), class_prediction)
    np.testing.assert_equal(acc.ground_truth_class(landmarks, len(slice_scores)), groundtruth_prediction)
    assert acc.volume(slice_scores, landmarks) == 0.8
    