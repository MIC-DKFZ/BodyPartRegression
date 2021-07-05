import torch

import numpy as np


class order_loss_h:
    """
    Heuristic order loss
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, scores_pred, z):
        scores_diff = scores_pred[:, 1:] - scores_pred[:, :-1]
        p_pred = torch.sigmoid(self.alpha * scores_diff)
        p_obs = torch.sigmoid(self.beta * z)
        loss = torch.mean((p_obs - p_pred) ** 2) 
        return loss


class order_loss_c:  
    """
    Classification order loss
    """

    def __init__(self):
        pass

    def __call__(self, scores_pred, _):
        scores_diff = scores_pred[:, 1:] - scores_pred[:, :-1]
        sigmoid = torch.sigmoid(scores_diff)
        # remove zeros --> they can lead to nan values
        sigmoid = sigmoid[sigmoid > 0]
        loss = -torch.mean(torch.log(sigmoid))
        return loss


class order_loss_c_plain: 
    """
    Classification order loss
    """

    def __init__(self):
        pass

    def __call__(self, scores_pred, _):
        scores_diff = scores_pred[:, 1:] - scores_pred[:, :-1]
        sigmoid = torch.sigmoid(scores_diff)
        loss = -torch.mean(torch.log(sigmoid))
        return loss


class no_order_loss:
    def __init__(self):
        pass

    def __call__(self, _, _):
        return 0.0
