import numpy as np 
from scipy import interpolate
from scipy.ndimage.measurements import mean

class NormalizedMSE: 
    def __init__(self): 
        pass

    def from_dataset(self, model, val_dataset, train_dataset): 
        val_score_matrix = model.compute_slice_score_matrix(val_dataset)
        train_score_matrix = model.compute_slice_score_matrix(train_dataset)
        d = self.get_normalizing_constant(np.nanmean(train_score_matrix, axis=0))
        mse, mse_std = self.from_matrices(val_score_matrix, train_score_matrix)
        
        return mse, mse_std, d


    def nmse_per_landmark_from_matrices(self, score_matrix, reference_matrix, d=False): 
        square_error_matrix = self.get_square_error_matrix(score_matrix, reference_matrix, d=d)
        nmse_per_landmark = np.nanmean(square_error_matrix, axis=0)

        counts = np.sum(np.where(~np.isnan(square_error_matrix), 1, 0), axis=0)
        nmse_errors = np.nanstd(square_error_matrix, ddof=1, axis=0)/np.sqrt(counts)

        return nmse_per_landmark, nmse_errors

    def nmse_per_volume_from_matrices(self, score_matrix, reference_matrix, d=False): 
        square_error_matrix = self.get_square_error_matrix(score_matrix, reference_matrix, d=d)
        nmse_per_volume =  np.nanmean(square_error_matrix, axis=1)
        counts = np.sum(np.where(~np.isnan(square_error_matrix), 1, 0), axis=1)
        nmse_errors = np.nanstd(square_error_matrix, ddof=1, axis=1)/np.sqrt(counts)

        return nmse_per_volume, nmse_errors
      


    def nmse_per_slice_from_matrices(self, score_matrix, reference_matrix, d=False): 
        square_error_matrix = self.get_square_error_matrix(score_matrix, reference_matrix, d=d)
        nmse =  np.nanmean(square_error_matrix)
        counts = np.sum(np.where(~np.isnan(square_error_matrix), 1, 0))
        nmse_std = np.nanstd(square_error_matrix, ddof=1)/np.sqrt(counts)

        return nmse, nmse_std 


    def from_matrices(self, score_matrix, reference_matrix, d=False):  ####### TODO ###### 
        nmses, nmse_stds = self.nmse_per_volume_from_matrices(score_matrix, reference_matrix, d=d)
        return np.mean(nmses), np.std(nmses, ddof=1)/np.sqrt(len(nmses))

        #square_error_matrix = self.get_square_error_matrix(score_matrix, reference_matrix, d=d)
        #nmse_per_volume =  np.nanmean(square_error_matrix, axis=1)

        #return np.mean(nmse_per_volume), np.std(nmse_per_volume, ddof=1)/ np.sqrt(len(nmse_per_volume))


    def get_square_error_matrix(self, score_matrix, reference_matrix, d=False): 
        expected_scores = np.nanmean(reference_matrix, axis=0) 
        if not d: d = np.abs(self.get_normalizing_constant(expected_scores))
        square_error_matrix = self.from_instance(expected_scores, score_matrix, d)
        return square_error_matrix

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
        return (expected_scores[-1] - expected_scores[0])/100
    
    # TODO 
    def volumes2MSEs(self, model, dataset, reference_dataset): 
        score_matrix = model.compute_slice_score_matrix(dataset)
        reference_score_matrix = model.compute_slice_score_matrix(reference_dataset)
        expected_scores = np.nanmean(reference_score_matrix, axis=0) 
        d = self.get_normalizing_constant(expected_scores) 
        mse_values = self.from_instance(expected_scores, score_matrix, d)
        dataset_ids = [dataset.landmark_ids[i] for i in np.arange(len(score_matrix), dtype=int)]

        return dataset_ids, mse_values 




