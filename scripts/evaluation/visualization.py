import matplotlib.pyplot as plt 
import numpy as np 



class Visualization: 
    def __init__(self): 
        pass

    def plot_landmarks(self, 
                       score_matrix, 
                       expected_scores, 
                       landmark_names, 
                       figsize=(16, 10), 
                       fontsize=16, 
                       text_margin_top=0.02, 
                       text_margin_right=2,
                       alpha=0.7,
                       ylim=None
                       ): 
        plt.figure(figsize=figsize)
        max_value = 0
        for idx in range(score_matrix.shape[1]): 
            x = score_matrix[:, idx]
            bins, _, _ = plt.hist(x, density=True, alpha=alpha, label=landmark_names[idx])
            mean = expected_scores[idx]
            plt.plot([mean, mean], [0, 100], color="black", linestyle="--", linewidth=0.5)
            
            plt.annotate(
                landmark_names[idx].replace("_", "-"),
                xy=(mean-text_margin_right, np.max(bins)+text_margin_top),
                fontsize= fontsize- 2,
            )
            
            if np.max(bins) > max_value: max_value = np.max(bins)

        if not ylim: plt.ylim((0, 1.1*max_value))
        else: plt.ylim(ylim)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        plt.xlabel("Slice Scores", fontsize=fontsize)
        plt.ylabel("Density Frequency Distribution", fontsize=fontsize)

