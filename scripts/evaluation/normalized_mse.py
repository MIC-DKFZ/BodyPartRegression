import numpy as np 
from scipy import interpolate

############ TODO ####################################### 
# mean landmarks -> nmse 

class NormalizedMSE: 
    def __init__(self): 
        pass

    def from_dataset(self, model, val_dataset, train_dataset): 
        val_score_matrix = model.compute_slice_score_matrix(val_dataset)
        train_score_matrix = model.compute_slice_score_matrix(train_dataset)
        mse, mse_std, d = self.from_matrices(val_score_matrix, train_score_matrix)
        
        return mse, mse_std, d


    def from_matrices(self, val_score_matrix, train_score_matrix, d=False): 
        expected_scores = np.nanmean(train_score_matrix, axis=0) 
        if not d: d = self.get_normalizing_constant(expected_scores) 
        square_error_matrix = self.from_instance(expected_scores, val_score_matrix, d)
        
        mean_square_error_per_landmark = np.nanmean(square_error_matrix, axis=0)
        mse = np.nanmean(mean_square_error_per_landmark)

        square_error_variance_per_landmark = np.nanvar(square_error_matrix, ddof=1, axis=0)
        mse_std = np.sqrt(np.sum(square_error_variance_per_landmark)/len(square_error_variance_per_landmark))

        return mse, mse_std, d


    def from_volume(self, landmarks, scores, expected_scores): 
        d = self.get_normalizing_constant(expected_scores) 
        expected_scores = expected_scores[~np.isnan(landmarks)]
        landmarks = np.array(landmarks[~np.isnan(landmarks)]).astype(int)
        scores = scores[landmarks]
        return self.from_instance(scores, expected_scores, d)

    def from_instance(self, y_truth, y_pred, d): 
        mse = ((y_truth - y_pred)/d)**2
        return mse 
    
    def get_normalizing_constant(self, expected_scores): 
        return expected_scores[-1] - expected_scores[0]

    def volumes2MSEs(self, model, dataset, reference_dataset): 
        score_matrix = model.compute_slice_score_matrix(dataset)
        reference_score_matrix = model.compute_slice_score_matrix(reference_dataset)
        expected_scores = np.nanmean(reference_score_matrix, axis=0) 
        d = self.get_normalizing_constant(expected_scores) 
        mse_values = self.from_instance(expected_scores, score_matrix, d)
        dataset_ids = [dataset.landmark_ids[i] for i in np.arange(len(score_matrix), dtype=int)]

        return dataset_ids, mse_values 




