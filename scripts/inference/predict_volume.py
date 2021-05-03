import albumentations as A 
import numpy as np 
import pickle
import os, sys
import torch 

sys.path.append("../../")
from scripts.network_architecture.bpr_model import BodyPartRegression

class PredictVolume: 
    """
    Predict the slice scores of a volume
    
    TODO: 
    - find valid region of slice scores 
        - output always: slice scores + slice index 
    """
    def __init__(self, base_dir, gpu=1):
        self.gpu = gpu
        self.load_model(base_dir)

    def load_model(self, base_dir): 
        self.base_filepath = base_dir
        self.config_filepath = base_dir + "config.p"
        self.model_filepath = base_dir + "model.pt"

        with open(self.config_filepath, "rb") as f: 
            self.config = pickle.load(f)
            
        self.model = BodyPartRegression(alpha=self.config["alpha"], 
                                        lr=self.config["lr"], 
                                        base_model=self.config["base_model"])
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()
        if self.gpu: 
            self.model.to("cuda")
        else: 
            self.model.to("cpu")
        
    def preprocess_npy(self, path: str, resize=False):
        x = np.load(path)
        if x.shape[0] == x.shape[1]: 
            return x.swapaxes(2, 1).swapaxes(1, 0)
        
        if resize: 
            transform = A.Compose([A.Resize(int(resize), int(resize))])
            x = transform(image=x)["image"]
            
        return x
    
    def predict_npy(self, path: str, resize=False): 
        x = self.preprocess_npy(path, resize=resize)
        x_tensor = torch.tensor(x[:, np.newaxis, :, :]).cuda()
        scores = self.predict_tensor(x_tensor)
        return scores, x
    
    
    def predict_tensor(self, tensor, n_splits=200): 
        scores = []
        n = tensor.shape[0]
        slice_splits = list(np.arange(0, n, n_splits)) 
        slice_splits.append(n)

        with torch.no_grad(): 
            self.model.eval() 
            for i in range(len(slice_splits) - 1): 
                min_index = slice_splits[i]
                max_index = slice_splits[i+1]
                score = self.model(tensor[min_index:max_index,:, :, :])
                scores += [s.item() for s in score]

        scores = np.array(scores)
        return scores

