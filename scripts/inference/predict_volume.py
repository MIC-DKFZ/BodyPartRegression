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
    def __init__(self, base_dir):
        self.load_model(base_dir)

    def load_model(self, base_dir): 
        self.base_filepath = base_dir
        self.config_filepath = base_dir + "config.p"
        self.model_filepath = base_dir + "model.pt"

        print(self.config_filepath)
        with open(self.config_filepath, "rb") as f: 
            print(f)
            self.config = pickle.load(f)
            
        self.model = BodyPartRegression(alpha=self.config["alpha"], 
                                        lr=self.config["lr"], 
                                        base_model=self.config["base_model"])
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()
        self.model.to("cuda")
        
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
    
    
    def predict_tensor(self, tensor): 
        if tensor.shape[0] <= 300: 
            with torch.no_grad(): 
                self.model.eval()
                scores = self.model(tensor) # TODO -> für zu viele slices aufsplitten
                scores = [y.item() for y in scores]
                
        elif tensor.shape[1] <= 600: 
            with torch.no_grad(): 
                self.model.eval()
                scores1 = self.model(tensor[0:300, :, :, :])
                scores1 =  [y.item() for y in scores1]

                scores2 = self.model(tensor[300:, :, :, :])
                scores2 =  [y.item() for y in scores2]
                scores = scores1 + scores2 

        elif tensor.shape[1] <= 900: 
            with torch.no_grad(): 
                self.model.eval()
                scores1 = self.model(tensor[0:300, :, :, :])
                scores1 =  [y.item() for y in scores1]

                scores2 = self.model(tensor[300:600, :, :, :])
                scores2 =  [y.item() for y in scores2]
                
                scores3 = self.model(tensor[600:, :, :, :])
                scores3 =  [y.item() for y in scores3]
                
                scores = scores1 + scores2 + scores3
        else: 
            with torch.no_grad(): 
                self.model.eval()
                scores1 = self.model(tensor[0:300, :, :, :])
                scores1 =  [y.item() for y in scores1]

                scores2 = self.model(tensor[300:600, :, :, :])
                scores2 =  [y.item() for y in scores2]

                scores3 = self.model(tensor[600:900, :, :, :])
                scores3 =  [y.item() for y in scores3]

                scores4 = self.model(tensor[900:, :, :, :])
                scores4 =  [y.item() for y in scores4]
                
                scores = scores1 + scores2 + scores3 + scores4
            
        scores = np.array(scores)
        return scores

