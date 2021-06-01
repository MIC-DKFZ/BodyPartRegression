import numpy as np 
from scipy import interpolate

class NormalizedMSE: 
    def __init__(self): 
        pass

    def from_dataset(self, model, val_dataset, train_dataset): 
        val_score_matrix = model.compute_slice_score_matrix(val_dataset)
        train_score_matrix = model.compute_slice_score_matrix(train_dataset)
        mse, mse_std, d = self.from_matrices(val_score_matrix, train_score_matrix)
        
        return mse, mse_std, d


    def from_matrices(self, val_score_matrix, train_score_matrix): 
        expected_scores = np.nanmean(train_score_matrix, axis=0) 
        d = expected_scores[-1] - expected_scores[0]
        mse_values = self.from_instance(expected_scores, val_score_matrix, d)
        mse = np.nanmean(mse_values)
        counts = np.sum(np.where(~np.isnan(mse_values), 1, 0))
        mse_std = np.nanstd(mse_values)/np.sqrt(counts)

        return mse, mse_std, d


    def from_volume(self, landmarks, scores, expected_scores): 
        d = expected_scores[-1] - expected_scores[0]
        expected_scores = expected_scores[~np.isnan(landmarks)]
        landmarks = np.array(landmarks[~np.isnan(landmarks)]).astype(int)
        scores = scores[landmarks]
        return self.from_instance(scores, expected_scores, d)

    def from_instance(self, y_truth, y_pred, d): 
        mse = ((y_truth - y_pred)/d)**2
        return mse 


