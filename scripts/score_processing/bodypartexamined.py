# TODO ####### # Write Tests for this class
import numpy as np 
    
class BodyPartExamined: 
    def __init__(self, classes, class2landmark, lookuptable):
        self.classes = classes
        self.class2landmark = class2landmark
        self.lookuptable = lookuptable
        self.class2score = self.get_class2score(class2landmark, lookuptable)


    def get_class2score(self, class2landmark, lookup): 
        i = 0
        for myClass, landmarks in class2landmark.items(): 
            for l in landmarks: 
                if isinstance(l, float) and np.isnan(l): 
                    if i == 0: 
                        class2score[myClass].append(-np.inf) 
                    else: 
                        class2score[myClass].append(np.inf)
                    continue
                class2score[myClass].append(lookup[l]["mean"])
            i += 1

    def body_range(self, slice_scores, lower_slice_index, upper_bound_score):
        body_region = np.where(slice_scores < upper_bound_score)[0]
        if len(body_region) == 0:
            return np.array([], dtype=int), 0
        upper_slice_index = np.max(body_region) + 1
        return np.arange(lower_slice_index, upper_slice_index), upper_slice_index

    def cut_scores(self, slice_scores, lower_slice_score, upper_slice_score): 
        score_window = np.where((slice_scores >= lower_slice_score) & (slice_scores < upper_slice_score))
        return score_window 

    def get_body_part_examined(self, slice_scores):
        body_part_examined = {bodypart: [] for bodypart in self.class2score}

        for bodypart, scores in self.class2score.items(): 
            body_range = self.cut_scores(slice_scores, scores[0], scores[1])
            body_part_examined[bodypart] = list(bodyrange.astype(float))

        return body_part_examined



def test_bodypartexamined(): 
    scores = [-10, -8.5, -7, -4, -1, 0, 2, 3, 5, 8, 9]
    bpe = {"1": [0, 1, 2, 3], 
            "2": [4, 5, 6], 
            "3": [7, 8, 9, 10]}

    lookup = {"l1": -8, "l2": -2, "l3": 3.5, "l4": 7}

    

if __name__ == "__main__": 
    classes = ["legs", "pelvis", "abdomen", "lungs", "shoulder-neck", "head"]
    class2landmark= {"legs": [np.nan, "pelvis-start"],
                     "pelvis": ["pelvis-start", "pelvis-end"], 
                     "abdomen": ["pelvis-end", "L1"], 
                     "chest": ["L1", "Th1"], 
                     "shoulder-neck": ["Th1", "C6"], 
                     "head": ["C6", np.nan]}
    bpe = BodyPartExamined()