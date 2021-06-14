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
from scripts.evaluation.normalized_mse import NormalizedMSE
from scripts.network_architecture.loss_functions import * 
from scripts.network_architecture.base_model import BodyPartRegressionBase

class SSBR(BodyPartRegressionBase):
    """[summary]
    SSBR model from Yan at. al with corrected classification order loss (no nan-values can be calculated). 
    Args:
        BodyPartRegressionBase ([type]): [description]
    """
    def __init__(self, 
                 lr=1e-4, 
                 alpha=0):

        super().__init__()
        BodyPartRegressionBase.__init__(self, lr=lr, lambda_=0, alpha=alpha, pretrained=False, 
                                   delta_z_max=np.inf, loss_order="c", beta_h=np.nan, 
                                   alpha_h=np.nan, weight_decay=0)

        # load vgg base model 
        self.conv6 = nn.Conv2d(512, 512, 1, stride=1, padding=0) # in_channel, out_channel, kernel_size
        self.fc7 = nn.Linear(512, 1)
        self.model = self.get_vgg()

        # overwrite order loss argument for SSBR model
        # self.loss_order = order_loss_c_plain()

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

    def loss(self, scores_pred, slice_indices, _): 
        loss_order =  self.loss_order(scores_pred, _) 
        ldist  =  self.alpha * self.loss_dist(scores_pred)
        loss = loss_order + ldist
        return loss, loss_order, ldist, 0

    def loss_dist(self, scores_pred): 
        scores_diff = (scores_pred[:, 1:]-scores_pred[:, :-1])
        loss = self.l1loss(scores_diff[:, 1:], scores_diff[:, :-1])
        return loss

