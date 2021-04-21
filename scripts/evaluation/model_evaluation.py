
import numpy as np 
import random
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm 

from itertools import cycle
import torch 



def grid_plot(X, titles=[], cols=4, rows=4, save_path=False, cmap="gray", vmax=200): 
    if rows > 3: 
        fig, axs = plt.subplots(rows, cols, figsize=(18, 16))

    elif rows == 3: 
        fig, axs = plt.subplots(rows, cols, figsize=(14, 15))
    else: 
        fig, axs = plt.subplots(rows, cols, figsize=(16, 10))

    idx = 0
    for row in range(rows): 
        for col in range(cols):
            axs[row, col].imshow(X[idx], cmap=cmap, vmin=0, vmax=vmax)
            if len(titles) == cols*rows: axs[row, col].set_title(titles[idx])
            axs[row, col].set_yticklabels([])
            axs[row, col].set_xticklabels([])
            idx += 1
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


class ModelEvaluation(): 
    def __init__(self): 
        pass

    def get_closest_slice(self, dataset, predict_dict, y_target, vol_idx): 
        y_pred = np.array(predict_dict[vol_idx])
        min_idx = np.argmin(np.abs(y_pred - y_target)).item()
        try: #TODO! 
            vol = dataset.preprocessed_volumes[vol_idx]
        except: 
            vol = dataset.get_full_volume(vol_idx)
        if len(vol.shape) == 4: closest_slice = vol[min_idx, 0, :, :]
        else:  closest_slice = vol[min_idx, :, :]
        closest_value = y_pred[min_idx]
        return closest_slice,  closest_value, min_idx

    def plot_closest_slices(self, dataset, predict_dict, y_target, skip_idx=[], skip_diff=3, save_path=False):
        vols = []
        titles = []
        # for vol_idx in dataset.preprocessed_volumes.keys(): 
        vol_indices = np.array(list(predict_dict.keys()))
        np.random.shuffle(vol_indices)

        for vol_idx in vol_indices: 
            if vol_idx in skip_idx: continue
            closest_slice, closest_value, min_idx = self.get_closest_slice(dataset, predict_dict, y_target, vol_idx)
            if np.abs(closest_value - y_target) < skip_diff: 
                vols.append(closest_slice)
                titles.append(f"Volume {vol_idx} closest image to score {y_target}")
        grid_plot(vols[0:9], titles[0:9], cols=3, rows=3, save_path=save_path)

    def plot_score_to_coronal_img(self, vol, scores, title="", targets=[-20, -10, 0, 10]): 
        fig = plt.figure(figsize=(8, 5), dpi= 80)
        img_coronal = np.flip(vol[:, 0, vol.shape[2]//2, :])
        revscores=np.array(scores)[::-1]
        
        ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((1, 4), (0, 3), sharey=ax1)

        ax1.imshow(img_coronal, cmap='gray', aspect='auto')
        ax2.plot(revscores, np.arange(len(revscores)))

        # Add a horizontal line
        colors = cycle(iter(['blue', 'red', 'green', 'orange', 'brown', 'yellow']))
        for target in targets:    
            idx = self.find_nearest(revscores, target)
            xyA = (target, idx)
            xyB = (0, idx)
            con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color=next(colors))
            ax2.add_artist(con)
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.title(title)
        plt.show()
        

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_pred_dict(self, model, dataset): 
        pred_dict = {}
        for key in tqdm(dataset.preprocessed_volumes.keys()): 
            x = dataset.preprocessed_volumes[key]
            x = torch.tensor(x)
            with torch.no_grad(): 
                model.eval()
                model.to("cpu")
                y = model(x)
            pred_dict[key] = y
        return pred_dict

    def score2slice_plot(self, pred_dict, filepath=None):
        plt.figure(figsize=(10, 12))
        for key in pred_dict.keys(): 
            plt.plot(pred_dict[key], label=f"Volume {key}")
        plt.legend(loc=0, fontsize=12)
        plt.xlabel("Slice index", fontsize=12)
        plt.ylabel("Predicted Score", fontsize=12)
        if isinstance(filepath, str): plt.savefig(filepath)
        plt.show()
        

    def initialize_landmark_dict(self, dataloader): 
        first_landmark_key = list(dataloader.dataset.landmark_dict.keys())[0]
        landmark_keys = dataloader.dataset.landmark_dict[first_landmark_key].keys()
        result_dict = {l: [] for l in landmark_keys}
        return result_dict

    def get_landmark_predictions(self, model, dataloader, max_eval_volumes=False): 
        pred_landmarks = self.initialize_landmark_dict(dataloader)
        random_scores = []

        if not max_eval_volumes: max_eval_volumes = len(dataloader.dataset.keys)

        with torch.no_grad(): 
            model.eval()
            model.to("cuda")

            for volume_key in dataloader.dataset.keys[0:max_eval_volumes]:   
                try: 
                    volume = dataloader.dataset.preprocessed_volumes[volume_key]
                except: 
                    volume = dataloader.get_full_volume(volume_key)
                volume = torch.tensor(volume).cuda()
                total_slices = volume.shape[0]
                for landmark_key in dataloader.dataset.landmark_dict[volume_key]: 

                    # get for volume and landmark the predicted score and save it to dictionary
                    slice_idx = dataloader.dataset.landmark_dict[volume_key][landmark_key]
                    landmark_score = self.get_slice_prediction(model, volume, [slice_idx])
                    if not np.isnan(landmark_score[0]): 
                        pred_landmarks[landmark_key] += landmark_score

                    # sample random numbers more often to get a more accurate estimate for the total variance
                    random_slice_indices = [random.randint(0, total_slices-1) for i in range(0, 5)]
                    random_score = self.get_slice_prediction(model, volume, random_slice_indices)
                    random_scores += random_score

            # calculate variance of data
            var_tot = np.nanvar(random_scores)

            # calculate variance for landmarks and mean predicted scores for landmarks
            landmarks_var_dict = {landmark: np.nanvar(pred_landmarks[landmark]) for landmark in pred_landmarks.keys()}
            landmarks_mean_dict = {landmark: np.nanmean(pred_landmarks[landmark]) for landmark in pred_landmarks.keys()}

        return landmarks_var_dict, landmarks_mean_dict, var_tot, pred_landmarks

    def get_slice_prediction(self, model, volume, slice_idx): 
        with torch.no_grad(): 
            model.eval()
            model.to("cuda")
            if np.isnan(slice_idx[0]): return [np.nan] 

            mySlice = volume[np.array(slice_idx), :, :, :]
            score = model(mySlice)
            score_list = [s.item() for s in score]
        return score_list

    def get_relative_var(self, landmarks_var_dict, var_tot): 
        rel_var_dict = {landmark: landmarks_var_dict[landmark]/var_tot for landmark in  landmarks_var_dict.keys()}
        rel_var_mean = np.nanmean(list(rel_var_dict.values()))
        rel_var_std = np.nanstd(list(rel_var_dict.values()))

        return rel_var_mean, rel_var_std, rel_var_dict

    def evaluate_landmark_error(self, model, dataloader, max_eval_volumes=False): 
        landmarks_var_dict, landmarks_mean_dict, var_tot, pred_landmarks = self.get_landmark_predictions(model, dataloader, 
                                                                                    max_eval_volumes=max_eval_volumes)
        rel_var_mean, rel_var_std, rel_var_dict = self.get_relative_var(landmarks_var_dict, var_tot)
        # print(rel_var_dict, rel_var_mean, rel_var_std, pred_landmarks)
        return rel_var_mean, rel_var_std, rel_var_dict 
