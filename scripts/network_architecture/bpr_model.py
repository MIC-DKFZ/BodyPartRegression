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
from scripts.evaluation.landmark_mse import LMSE
from scripts.network_architecture.loss_functions import * 
from scripts.network_architecture.base_model import BodyPartRegressionBase

class BodyPartRegression(BodyPartRegressionBase):
    def __init__(self, 
                 lr=1e-4, 
                 lambda_=0, 
                 alpha=0,
                 pretrained=False, 
                 delta_z_max = np.inf,
                 loss_order="h", 
                 beta_h=0.025, 
                 alpha_h=0.5,
                 base_model="vgg", # TODO löschen
                 weight_decay=0):

        BodyPartRegressionBase.__init__(self, lr=lr, lambda_=lambda_, alpha=alpha, pretrained=pretrained, 
                                   delta_z_max=delta_z_max, loss_order=loss_order, beta_h=beta_h, 
                                   alpha_h=alpha_h, weight_decay=weight_decay)
        # load vgg base model 
        self.conv6 = nn.Conv2d(512, 512, 1, stride=1, padding=0) # in_channel, out_channel, kernel_size
        self.fc7 = nn.Linear(512, 1)
        self.model = self.get_vgg()

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
