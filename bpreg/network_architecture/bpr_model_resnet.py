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
import datetime
import random, sys
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

cv2.setNumThreads(1)

sys.path.append("../../")
from bpreg.evaluation.landmark_mse import LMSE
from bpreg.network_architecture.loss_functions import *
from bpreg.network_architecture.base_model import BodyPartRegressionBase


class BodyPartRegressionResNet(BodyPartRegressionBase):
    def __init__(
        self,
        lr: float = 1e-4,
        lambda_: float = 0,
        alpha: float = 0,
        pretrained: bool = False,
        delta_z_max: float = np.inf,
        loss_order: str = "h",
        beta_h: float = 0.025,
        alpha_h: float = 0.5,
        weight_decay: float = 0,
    ):

        BodyPartRegressionBase.__init__(
            self,
            lr=lr,
            lambda_=lambda_,
            alpha=alpha,
            pretrained=pretrained,
            delta_z_max=delta_z_max,
            loss_order=loss_order,
            beta_h=beta_h,
            alpha_h=alpha_h,
            weight_decay=weight_decay,
        )

        # load resnet base model
        self.fc7_res = nn.Linear(2048, 1)
        self.model = self.get_resnet()

    def get_resnet(self):
        resnet50 = models.resnet50(pretrained=self.pretrained)
        resnet50.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
        return resnet50

    def forward(self, x: torch.Tensor):
        x = F.relu(self.model(x.float()))
        x = x.view(-1, 2048)
        x = self.fc7_res(x)
        return x
