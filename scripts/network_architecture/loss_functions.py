import torch

class loss_order_h: 
    """
    Heuristic order loss
    """
    def __init__(self, alpha, beta): 
        self.alpha = alpha
        self.beta = beta
    def __call__(self, scores_pred, z): 
        scores_diff = scores_pred[:, 1:] - scores_pred[:, :-1]
        p_pred = torch.sigmoid(self.alpha*scores_diff)
        p_obs = torch.sigmoid(self.beta*z)
        loss = torch.mean((p_obs - p_pred)**2) #TODO *6! 
        return loss

class loss_order_c: #TODO add 
    """
    Classification order loss
    """
    def __init__(self): 
        pass
    def __call__(self, scores_pred, _): 
        scores_diff = scores_pred[:, 1:] - scores_pred[:, :-1]
        loss = - torch.mean(torch.log(torch.sigmoid(scores_diff)))
        return loss