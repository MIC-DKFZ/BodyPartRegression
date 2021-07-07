"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import random, sys
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

cv2.setNumThreads(1)

sys.path.append("../../")
from bpreg.network_architecture.loss_functions import *
from bpreg.network_architecture.base_model import BodyPartRegressionBase


class SSBR(BodyPartRegressionBase):
    """[summary]
    SSBR model from Yan at. al with corrected classification order loss (no nan-values can be calculated).
    Args:
        BodyPartRegressionBase ([type]): [description]
    """

    def __init__(self, lr=1e-4, alpha=0):

        super().__init__()
        BodyPartRegressionBase.__init__(
            self,
            lr=lr,
            lambda_=0,
            alpha=alpha,
            pretrained=False,
            delta_z_max=np.inf,
            loss_order="c",
            beta_h=np.nan,
            alpha_h=np.nan,
            weight_decay=0,
        )

        # load vgg base model
        self.conv6 = nn.Conv2d(
            512, 512, 1, stride=1, padding=0
        )  # in_channel, out_channel, kernel_size
        self.fc7 = nn.Linear(512, 1)
        self.model = self.get_vgg()

    def get_vgg(self):
        vgg16 = models.vgg16(pretrained=self.pretrained)
        vgg16.features[0] = torch.nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

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
        loss_order = self.loss_order(scores_pred, _)
        ldist = self.alpha * self.loss_dist(scores_pred)
        loss = loss_order + ldist
        return loss, loss_order, ldist, 0

    def loss_dist(self, scores_pred):
        scores_diff = scores_pred[:, 1:] - scores_pred[:, :-1]
        loss = self.l1loss(scores_diff[:, 1:], scores_diff[:, :-1])
        return loss
