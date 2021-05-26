import numpy as np 
import datetime
import random, sys
import torch
import cv2
import albumentations as A
from scipy.stats import pearsonr
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

cv2.setNumThreads(1)

sys.path.append("../../")


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

class BodyPartRegression(pl.LightningModule):
    """
    Architecture of the BodyPartRegression Model based on Yan et. als Self-Supervised Body Part Regression model (SSBR). 
    # TODO Cite 
    """
    def __init__(self, 
                 lr=1e-4, 
                 lambda_=0, 
                 alpha=0,
                 pretrained=False, 
                 delta_z_max = np.inf,
                 loss_order="h", 
                 beta_h=0.025, 
                 alpha_h=0.5,
                 base_model="vgg", 
                 weight_decay=0):

        super().__init__()
        self.lr = lr 
        self.alpha_h = alpha_h
        self.beta_h = beta_h
        self.alpha = alpha 
        self.weight_decay=weight_decay
        self.loss_order_name = loss_order
        self.delta_z_max = delta_z_max
        self.l1loss = torch.nn.SmoothL1Loss(reduction="mean")
        self.pretrained = pretrained
        self.conv6 = nn.Conv2d(512, 512, 1, stride=1, padding=0) # in_channel, out_channel, kernel_size
        self.fc7 = nn.Linear(512, 1)
        self.lambda_ = lambda_
        self.val_landmark_metric = []
        self.val_loss = []
        self.base_model = base_model
        self.hparams = {"alpha": alpha, "lambda": lambda_, 
                        "loss_order": loss_order, "beta_h": beta_h, 
                        "alpha_h": alpha_h, "lr": lr}

        self.model = self.get_vgg()
        
        if loss_order == "h": 
            self.loss_order = loss_order_h(alpha=self.alpha_h, beta=self.beta_h)
        elif loss_order == "c": 
            self.loss_order = loss_order_c()
        else: raise ValueError(f"Unknown loss parameter {loss_order}")




    def get_vgg(self):
        vgg16 = models.vgg16(pretrained=self.pretrained)
        vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vgg16.to(device)

        return vgg16.features

    def forward(self, x: torch.Tensor):
        x = self.model(x.float())
        x = F.relu(self.conv6(x))
        x = torch.mean(x, axis=(2, 3))
        x = x.view(-1, 512)
        x = self.fc7(x)
        return x

    def base_step(self, batch, batch_idx): 
        x, slice_indices, z = batch
        x, batch_size, num_slices = self.to1channel(x)
        y_hat = self(x)
        y_hat = self.tonchannel(y_hat, batch_size, num_slices)
        loss, loss_order, loss_dist, loss_l2 = self.loss(y_hat, slice_indices, z)
        return loss, loss_order, loss_dist, loss_l2
    
    def training_step(self, batch, batch_idx):
        loss, loss_order, loss_dist, loss_l2 = self.base_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_loss_order', loss_order)
        self.log('train_loss_dist', loss_dist)
        self.log('train_loss_l2', loss_l2)
        return loss
    
    def to1channel(self, x): 
        batch_size  = x.shape[0]
        num_slices = x.shape[1]
        x = x.reshape(batch_size * num_slices, 1, x.shape[2], x.shape[3])
        return x, batch_size, num_slices
    
    def tonchannel(self, x, batch_size, num_slices): 
        x = x.reshape(batch_size, num_slices)
        return x
        
    def validation_epoch_end(self, validation_step_outputs): 
        val_dataloader = self.val_dataloader()
        train_dataloader = self.train_dataloader()

        landmark_mean, landmark_var, total_var = self.landmark_metric(val_dataloader.dataset)
        mse_t, mse_t_std, d = self.normalized_mse(val_dataloader.dataset, train_dataloader.dataset)
        mse_v, mse_v_std, d = self.normalized_mse(val_dataloader.dataset, val_dataloader.dataset)

        self.log('val_landmark_metric_mean', landmark_mean)
        self.log('val_landmark_metric_var', landmark_var)
        self.log('total variance', total_var)
        self.log('mse_t', mse_t)
        self.log('mse_v', mse_v)

        self.log('d', d)
      
    def validation_step(self, batch, batch_idx):
        loss, loss_order, loss_dist, loss_l2 = self.base_step(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_loss_order', loss_order)
        self.log('val_loss_dist', loss_dist)
        self.log('val_loss_l2', loss_l2)



    def test_step(self, batch, batch_idx):
        loss, loss_order, loss_dist, loss_l2 = self.base_step(batch, batch_idx)
        dataloader = self.test_dataloader()
        landmark_mean, landmark_var, total_var = self.landmark_metric(dataloader.dataset)
        
        self.log('test_loss', loss)
        self.log('test_loss_order', loss_order)
        self.log('test_loss_dist', loss_dist)
        self.log('test_loss_l2', loss_l2)
        self.log('test_landmark_metric_mean', landmark_mean)
        self.log('test_landmark_metric_var', landmark_var)
     
        if len(self.val_loss) > 6: 
            x = np.array(self.val_loss)
            y = np.array(self.val_landmark_metric)
            indices = np.where((x != np.inf) &(x != np.nan))[0]
            pcorr = pearsonr(x[indices], y[indices])
            self.log('pearson-correlation', torch.tensor(pcorr[0]))    
            
    def loss(self, scores_pred, slice_indices, z): 
        l2_norm = 0 
        ldist_reg = 0
        loss_order = self.loss_order(scores_pred, z) 
        if self.lambda_ > 0: l2_norm = self.lambda_ * torch.mean(scores_pred**2)
        if self.alpha > 0: ldist_reg = self.alpha * self.loss_dist(scores_pred, z)
        loss = loss_order + l2_norm + ldist_reg
        return loss, loss_order, ldist_reg, l2_norm
    
    def loss_dist(self, scores_pred, z): 
        mask = torch.where(z > self.delta_z_max, 0, 1)
        scores_diff = (scores_pred[:, 1:]-scores_pred[:, :-1])*mask
        loss = self.l1loss(scores_diff[:, 1:], scores_diff[:, :-1])
        return loss
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def landmark_metric(self, dataset): 
        slice_score_matrix = self.compute_slice_score_matrix(dataset)
        landmark_vars = np.nanvar(slice_score_matrix, axis=0)
        total_var = np.nanvar(slice_score_matrix)
        return np.nanmean(landmark_vars/total_var), np.nanstd(landmark_vars/total_var), total_var
    

    def compute_slice_score_matrix(self, dataset): 
        with torch.no_grad(): 
            self.eval() 
            self.to("cuda")
            slice_score_matrix = np.full(dataset.landmark_matrix.shape, np.nan)

            for i, slices, defined_landmarks in zip(np.arange(0, slice_score_matrix.shape[0]), 
                                                    dataset.landmark_slices_per_volume,
                                                    dataset.defined_landmarks_per_volume): 
                scores = self(torch.tensor(slices[:, np.newaxis, :, :]).cuda())
                slice_score_matrix[i, defined_landmarks] = scores[:, 0].cpu().detach().numpy()
        return slice_score_matrix

    def normalized_mse(self, val_dataset, train_dataset): 
        val_score_matrix = self.compute_slice_score_matrix(val_dataset)
        train_score_matrix = self.compute_slice_score_matrix(train_dataset)
        mse, mse_std, d = normalized_mse_from_matrices(val_score_matrix, train_score_matrix)
        
        return mse, mse_std, d


def normalized_mse_from_matrices(val_score_matrix, train_score_matrix): 
    expected_slice_scores = np.nanmean(train_score_matrix, axis=0) 
    d = (expected_slice_scores[-1] - expected_slice_scores[0]) 

    mse_values = ((val_score_matrix - expected_slice_scores)/d)**2
    mse = np.nanmean(mse_values)
    counts = np.sum(np.where(~np.isnan(mse_values), 1, 0))
    mse_std = np.nanstd(mse_values)/np.sqrt(counts)

    return mse, mse_std, d









############################## TODO ################################################################################
"""
    def get_resnet(self): 
        resnet50 = models.resnet50(pretrained=self.pretrained)
        resnet50.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1])) 
        return resnet50

    def forward_res(self, x:torch.Tensor): 
        x = F.relu(self.model(x.float()))
        x = x.view(-1, 1000)
        x = self.fc7_res(x)
        return x
"""
