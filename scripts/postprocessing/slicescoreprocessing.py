import torch
import numpy as np 
import sys, os

sys.path.append("../../")
from scripts.inference.predict_volume import PredictVolume

class SliceScoreProcessing(PredictVolume): 
    def __init__(self, base_dir, gpu=1): 
        self.base_dir = base_dir
        PredictVolume.__init__(self, base_dir, gpu=gpu)
    
    def cut_window(self, y: np.array, min_value: int, max_value: int):
        smaller_min_cut = np.where(y < min_value)[0]
        greater_max_cut = np.where(y > max_value)[0]

        if len(smaller_min_cut) == 0: min_cut = 0
        else: min_cut = smaller_min_cut[-1] + 1

        if len(greater_max_cut) == 0: max_cut = len(y)
        else: max_cut = greater_max_cut[0]
        return np.arange(min_cut, max_cut)
    
    def cut_mask(self, filepath_source, filepath_mask, min_cut, max_cut): 
        # load mask
        mask, _ = self.load_nii(filepath_mask, swapaxes=1)

        # load scores 
        scores, x = self.predict_nii(filepath_source)

        # get cut-indices
        indices_valid = self.cut_window(scores, min_cut, max_cut)

        return indices_valid, scores, mask, x

    def check_false_positives(self, segmentation, indices_valid, filename="", printMe=False): 
        segmentation_zindices = np.where(segmentation>0)[0]
        if len(indices_valid) == 0: 
            if len(segmentation_zindices) > 0: return 1
            return 0 
        if (min(segmentation_zindices) < min(indices_valid)) or (max(segmentation_zindices) > max(indices_valid)): 
            if printMe: 
                print(f"False Positive {filename}")
                print(f"Valid range for indices: {min(indices_valid)} - {max(indices_valid)}")
                print(f"Indices of segmentation: {np.unique(segmentation_zindices)}\n")
            return 1
        return 0
    
    def get_bound_lists(self, data_path): 
        upper_bound_list = []
        lower_bound_list = []

        for file in tqdm(os.listdir(data_source_path)): 
            try: scores, _ = bpr.predict_nii(data_source_path + file)
            except: print(file); continue;
            if len(scores) == 0: continue
                
            # Take 2% percentile as minimum - for more robustness to outliers
            minimum = np.percentile(scores, 2)
            # Take 98% as maximum - for more robustness to outliers
            maximum = np.percentile(scores, 98)
            
            minimum = np.min(scores) # TODO ! 
            maximum = np.max(scores)

            lower_bound_list.append(minimum)
            upper_bound_list.append(maximum)     
        return lower_bound_list, upper_bound_list

    def find_min_max_cut(self, data_path): 
        lower_bound_list, upper_bound_list = self.get_bound_lists(data_path)

        # use 25% and 75% percentile for cuts
        max_cut = np.percentile(upper_bound_list, 75)
        min_cut = np.percentile(lower_bound_list, 25)

        return min_cut, max_cut


    def slice_score_postprocessing(self, filepaths_mask, filepaths_source, min_cut, max_cut, func = lambda x: np.where(x > 0, 1, 0)): 
        myDict = {}
        for filepath_source, filepath_mask in tqdm(zip(filepaths_source, filepaths_mask)): 
            filename_mask = filepath_mask.split("/")[-1]
            indices_valid, scores, x_mask, x = self.cut_mask(filepath_source, filepath_mask, min_cut, max_cut)
            segmentation =  func(x_mask) #np.where(x_mask > 0 , 1, 0)
            if np.sum(segmentation) == 0: continue
            fp = self.check_false_positives(segmentation, indices_valid, filename=filepath_mask, printMe=False)
            myDict[filename_mask] = {}
            myDict[filename_mask]["valid-indices"] = indices_valid
            myDict[filename_mask]["segmentation-indices"] = np.unique(np.where(segmentation>0)[0])
            myDict[filename_mask]["filepath"] = filepath_mask
            myDict[filename_mask]["false positive"] = fp 


        return myDict 

    def print_dict(self, myDict, subset=[]):
        if len(subset) == 0: subset = myDict.keys()

        for key, myDict in myDict.items(): 
            if not key in subset: continue
            if len(myDict['valid-indices']) == 0: myDict['valid-indices'] = [-1]
            print(key)
            print(f"valid region:\t\t{min(myDict['valid-indices'])} - {max(myDict['valid-indices'])}")
            print(f"Segmentation range:\t{min(myDict['segmentation-indices'])} - {max(myDict['segmentation-indices'])}\n")

    def summary_fp(self, myDict, fp_groundtruth_filenames): 
        catched_false_positives = [key for key in myDict.keys() if myDict[key]["false positive"] == 1]
        uncatched_fps = list(set(fp_groundtruth_filenames) - set(catched_false_positives))
        incorrect_fps = list(set(catched_false_positives) - set(fp_groundtruth_filenames))

        print(f"Ground truth false positives: {len(fp_groundtruth_filenames)}")
        print(f"Catched false positives: {len(catched_false_positives) }\n")
        if len(uncatched_fps) > 0: 
            print("Uncatched false positives: ")
            self.print_dict(myDict, subset=np.sort(uncatched_fps))

        else: 
            print("All false positives has been catched")

        if len(incorrect_fps) > 0: 
            print("Wrong catched files: ")
            self.print_dict(myDict, subset=np.sort(incorrect_fps))
        else: 
            print("All catched files are correct. ")

        print(f"Accuracy: {((len(catched_false_positives) - len(incorrect_fps))*100/len(fp_groundtruth_filenames)):1.0f}%")

    def fp_analysis(self, filepaths_mask, filepaths_source, fp_groundtruth_filenames): 
        fp_dict = self.slice_score_postprocessing(filepaths_mask, filepaths_source)
        self.summary_fp(fp_dict, fp_groundtruth_filenames)

        return fp_dict 

    
