import pickle
import numpy as np 

sys.path.append("../../")
from scripts.score_processing.datasanitychecks import DataSanityCheck
from scripts.preprocessing.nifti2npy import Nifti2Npy
from scripts.model_architecture.bpr_model import BodyPartRegression


# TODO Use InferenceModel in modelEvaluation 
# TODO put prediction code from bpr_model --> InferenceModel 
# TODO convert Dockercontainer again and starter.py
# TODO remove predict.py
# TODO Use InferenceModel for SliceScoreProcessing 
# TODO Save in InferenceModel Lookup Table for all defined landmarks from val. dataset 
# TODO Use nifti2npy for preprocessing 
# TODO Write Tests for scores2bodyrange 

class InferenceModel: 
    def __init__(self, base_dir, device="cuda"): 
        self.landmark_matrix = ""
        self.landmark_files = ""
        self.lookup = ""
        self.device = device
        self.model = self.load_model(base_dir)
        self.n2n = Nifti2Npy(
            target_pixel_spacing=3.5, min_hu=-1000, max_hu=1500, size=128
        )


    def load_model(self, 
                   base_dir, 
                   model_file="model.pt", 
                   config_file="config.p"): 
        self.base_filepath = base_dir
        self.config_filepath = base_dir + config_file # TODO
        self.model_filepath = base_dir + model_file # TODO verallgemeinern

        with open(self.config_filepath, "rb") as f: 
            self.config = pickle.load(f)
            
        self.model = BodyPartRegression(alpha=self.config["alpha"], 
                                        lr=self.config["lr"], 
                                        base_model=self.config["base_model"])
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()
        self.model.to(self.device)


    def predict_tensor(self, tensor, n_splits=200): 
        scores = []
        n = tensor.shape[0]
        slice_splits = list(np.arange(0, n, n_splits)) 
        slice_splits.append(n)

        with torch.no_grad(): 
            self.eval() 
            self.to(inference_device)
            for i in range(len(slice_splits) - 1): 
                min_index = slice_splits[i]
                max_index = slice_splits[i+1]
                score = self(tensor[min_index:max_index,:, :, :].to(self.device))
                scores += [s.item() for s in score]

        scores = np.array(scores)
        return scores

    def predict_npy(self, x, n_splits=200): 
        x_tensor = torch.tensor(x[:, np.newaxis, :, :]).to(self.device)
        scores = self.predict_tensor(x_tensor, n_splits=n_splits)
        return scores

    def predict_nifti(self): 
        # get nifti file as tensor
        x, pixel_spacings = self.n2n.preprocess_nifti(nifti_path)
        x = np.transpose(x, (2, 0, 1))[:, np.newaxis, :, :]
        x_tensor = torch.tensor(x)
        x_tensor.to(self.device)

        # predict slice-scores
        scores = self.predict_tensor(x_tensor)
        return scores, pixel_spacings      

    def scores2bodyrange(): 
        pass

    def datasanitychecks(): 
        pass 

    def predict_nifti2json(self,  nifti_path, output_path): 
        scores, pixel_spacings = self.predict_nifti(nifti_path)

        # validation
        dsc = DataSanityCheck(
            scores,
            pixel_spacings[2],
            self.slope_mean,
            self.slope_std,
            self.lower_bound_score,
            self.upper_bound_score,
            self.smoothing_sigma,
        )
        cleaned_scores, valid_indices = dsc.remove_invalid_regions(scores)
        reverse_zordering = dsc.is_reverse_zordering()
        valid_zspacing = dsc.is_valid_zspacing()

        # estimate BodyPartExamined
        body_part_examined = self._get_body_part_examined(cleaned_scores)

        # transform to 0-100
        cleaned_scores = self._transform_0_100(cleaned_scores)
        scores = self._transform_0_100(scores)
        lookup_transformed = self._transform_lookup()

        # write output json file
        json_output = {
            "slice scores": list(cleaned_scores),
            "valid indices": list(valid_indices.astype(float)),
            "unprocessed slice scores": list(scores),
            "body part examined": body_part_examined,
            "look-up table": lookup_transformed,
            "reverse z-ordering": reverse_zordering,
            "valid z-spacing": valid_zspacing,
            "z-spacing": np.round(float(pixel_spacings[2]), 2),
            "expected z-spacig": np.round(dsc.zhat, 2),
        }
        
        with open(output_path, "w") as f:
            json.dump(json_output, f)

        return json_output