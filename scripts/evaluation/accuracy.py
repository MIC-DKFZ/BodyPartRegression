import numpy as np 

class Accuracy: 
    def __init__(self, 
                 estimated_landmark_scores, 
                 class_to_landmark): 
        self.estimated_landmark_scores = estimated_landmark_scores
        self.class_to_landmark = class_to_landmark
        self.class_to_score_mapping = self.get_class_to_score_mapping()
        
    def volume(self, scores, landmark_positions): 
        _, accuracies = self.slice_accuracies(scores, landmark_positions)
        if len(accuracies) == 0: return np.nan
        
        return np.mean(accuracies)
    
    def slice_accuracies(self, scores, landmark_positions): 
        ground_truth_classes = self.ground_truth_class(landmark_positions, max_slices=len(scores))
        predicted_classes = self.class_prediction(scores)
        
        indices = np.where(~np.isnan(ground_truth_classes)) 
        ground_truth_classes = ground_truth_classes[indices]
        predicted_classes = predicted_classes[indices]

        accuracies = np.array((ground_truth_classes == predicted_classes)*1)

        return indices[0], accuracies

    def get_class_to_score_mapping(self): 
        class_to_score_mapping = {myClass: [] for myClass in self.class_to_landmark.keys()}
        
        # iterate through classes 
        for myClass in self.class_to_landmark: 
            # get lan
            landmarks = self.class_to_landmark[myClass]
            slice_scores = self.estimated_landmark_scores[landmarks]
            class_to_score_mapping[myClass] = list(slice_scores)
            
        return class_to_score_mapping
    
    def class_prediction(self, slice_scores): 
        class_prediction = np.full(slice_scores.shape, np.nan) 
        
        # iterate through classes
        for myClass in self.class_to_score_mapping.keys(): 
            min_score = self.class_to_score_mapping[myClass][0]
            max_score = self.class_to_score_mapping[myClass][1]
            
            # set class, if slice score is in between slice score range 
            class_prediction = np.where((slice_scores >= min_score) & 
                                        (slice_scores < max_score), myClass, class_prediction)
            
        return class_prediction
    
    def ground_truth_class(self, landmark_positions, max_slices): 
        classes = np.full((max_slices), np.nan)
        max_class = np.max(list(self.class_to_landmark.keys()))
        min_class = np.min(list(self.class_to_landmark.keys()))
        first_ground_truth_class = np.nan 
        for myClass, landmarks in self.class_to_landmark.items(): 
            # get start and end slice index position of class
            positions = landmark_positions[landmarks]

            # if class start or end is not defined --> skip 
            if np.isnan(positions[0]) or np.isnan(positions[1]): continue

            # get slice indices for start and end of class
            class_indices = np.arange(positions[0], positions[1], dtype=int)

            # set class for slice index range of class 
            classes[class_indices] = myClass
            if np.isnan(first_ground_truth_class): first_ground_truth_class = myClass

        return classes