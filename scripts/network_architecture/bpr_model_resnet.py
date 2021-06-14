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

class BodyPartRegressionResNet(BodyPartRegressionBase):
    def __init__(self, 
                 lr=1e-4, 
                 lambda_=0, 
                 alpha=0,
                 pretrained=False, 
                 delta_z_max = np.inf,
                 loss_order="h", 
                 beta_h=0.025, 
                 alpha_h=0.5,
                 weight_decay=0):

        BodyPartRegressionBase.__init__(self, lr=lr, lambda_=lambda_, alpha=alpha, pretrained=pretrained, 
                                   delta_z_max=delta_z_max, loss_order=loss_order, beta_h=beta_h, 
                                   alpha_h=alpha_h, weight_decay=weight_decay)
                                   
        # load resnet base model 
        self.fc7_res = nn.Linear(2048, 1)
        self.model = self.get_resnet()

    def get_resnet(self): 
        resnet50 = models.resnet50(pretrained=self.pretrained)
        resnet50.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1])) 
        return resnet50

    def forward(self, x:torch.Tensor): 
        x = F.relu(self.model(x.float()))
        x = x.view(-1, 2048)
        x = self.fc7_res(x)
        return x
