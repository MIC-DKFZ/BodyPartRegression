from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np 


class ValidationVolume: 
    def __init__(self, model, val_dataset, idx, expected_scores): 
        self.landmark_idx = idx
        self.data_idx = val_dataset.landmark_ids[idx]
        
        self.landmarks = val_dataset.landmark_matrix[self.landmark_idx, :]
        self.landmark_names = val_dataset.landmark_names 

        self.nonempty = np.where(~np.isnan(self.landmarks))[0]
        self.filename =  val_dataset.filenames[self.data_idx]
        self.expected_scores = expected_scores
        self.X = val_dataset.get_full_volume(self.data_idx)
        self.z = val_dataset.z_spacings[self.data_idx]
        self.scores = model.predict_npy(self.X)
        self.interpolated_x, self.interpolated_scores = self.get_interpolated_scores(self.landmarks, 
                                                                self.expected_scores)
        self.z_array = np.arange(0, len(self.X))*self.z
        self.fontsize = 16
        self.figsize=(14, 8)
        self.markerstyles = np.array([".", "v", "*", "p", "D", "X", "h", "P", "+", "^", "x", "d"])
        
    def plot_scores(self, set_figsize=False):
        if set_figsize: plt.figure(figsize=self.figsize)
        
        # plot scores 
        plt.plot(self.z_array, self.scores)
        
        # plot landmarks
        for i, name, landmark, score in zip(np.arange(len(self.nonempty)),
                                       self.landmark_names[self.nonempty],
                                       self.landmarks[self.nonempty], 
                                       self.scores[self.landmarks[self.nonempty].astype(int)]): 
            plt.plot(landmark*self.z, 
                     score,
                     linestyle="", 
                     marker=self.markerstyles[i],
                     color="black", 
                     label=name)
        
        plt.xlabel("$\Delta$z [mm]", fontsize=self.fontsize)
        plt.ylabel("Score", fontsize=self.fontsize)
        plt.xticks(fontsize=self.fontsize-2)
        plt.yticks(fontsize=self.fontsize-2)

    
    def plot_interpolated_scores(self): 
        self.plot_scores(set_figsize=True)
        plt.plot(self.interpolated_x*self.z, self.interpolated_scores, linestyle="--", linewidth=2)

    
    def get_interpolated_scores(self, landmarks, expected_scores): 
        nonempty = np.where(~np.isnan(landmarks))
        x = np.arange(np.nanmin(landmarks), np.nanmax(landmarks) + 1)
        
        func = interpolate.interp1d(landmarks[nonempty], expected_scores[nonempty], kind="linear")
        interpolated_scores = func(x)
        
        return x, interpolated_scores
        
