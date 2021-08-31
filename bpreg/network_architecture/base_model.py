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
import albumentations as A
from scipy.stats import pearsonr
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

cv2.setNumThreads(1)

sys.path.append("../../")
from bpreg.evaluation.landmark_mse import LMSE
from bpreg.network_architecture.loss_functions import *


class BodyPartRegressionBase(pl.LightningModule):
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

        super().__init__()
        self.lr = lr
        self.alpha_h = alpha_h
        self.beta_h = beta_h
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.loss_order_name = loss_order
        self.delta_z_max = delta_z_max
        self.l1loss = torch.nn.SmoothL1Loss(reduction="mean")
        self.pretrained = pretrained
        self.lambda_ = lambda_
        self.val_landmark_metric = []
        self.val_loss = []
        self.save_hyperparameters()
        self.lmse = LMSE()

        if loss_order == "h":
            self.loss_order = order_loss_h(alpha=self.alpha_h, beta=self.beta_h)
        elif loss_order == "c":
            self.loss_order = order_loss_c()

        elif loss_order == "":
            self.loss_order = no_order_loss()
        else:
            raise ValueError(f"Unknown loss parameter {loss_order}")

    def base_step(self, batch, batch_idx):
        x, slice_indices, z = batch
        x, batch_size, num_slices = self.to1channel(x)
        y_hat = self(x)
        y_hat = self.tonchannel(y_hat, batch_size, num_slices)
        loss, loss_order, loss_dist, loss_l2 = self.loss(y_hat, slice_indices, z)
        return loss, loss_order, loss_dist, loss_l2

    def training_step(self, batch, batch_idx):
        loss, loss_order, loss_dist, loss_l2 = self.base_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_loss_order", loss_order)
        self.log("train_loss_dist", loss_dist)
        self.log("train_loss_l2", loss_l2)
        return loss

    def to1channel(self, x):
        batch_size = x.shape[0]
        num_slices = x.shape[1]
        x = x.reshape(batch_size * num_slices, 1, x.shape[2], x.shape[3])
        return x, batch_size, num_slices

    def tonchannel(self, x, batch_size, num_slices):
        x = x.reshape(batch_size, num_slices)
        return x

    def validation_epoch_end(self, validation_step_outputs):
        val_dataloader = self.val_dataloader()
        train_dataloader = self.train_dataloader()

        lmse, lmse_std, d = self.lmse.from_dataset(
            self, val_dataloader.dataset, train_dataloader.dataset
        )

        self.log("lmse", lmse)
        self.log("lmse_std", lmse_std)
        self.log("d", d)

    def validation_step(self, batch, batch_idx):
        loss, loss_order, loss_dist, loss_l2 = self.base_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_loss_order", loss_order)
        self.log("val_loss_dist", loss_dist)
        self.log("val_loss_l2", loss_l2)

    def test_step(self, batch, batch_idx):
        test_dataloader = self.test_dataloader()
        train_dataloader = self.train_dataloader()

        lmse, lmse_std, d = self.lmse.from_dataset(
            self, test_dataloader.dataset, train_dataloader.dataset
        )

        self.log("lmse", lmse)
        self.log("lmse_std", lmse_std)
        self.log("d", d)

    def loss(self, scores_pred, slice_indices, z):
        l2_norm = 0
        ldist_reg = 0
        loss_order = self.loss_order(scores_pred, z)
        if self.lambda_ > 0:
            l2_norm = self.lambda_ * torch.mean(scores_pred ** 2)
        if self.alpha > 0:
            ldist_reg = self.alpha * self.loss_dist(scores_pred, z)
        loss = loss_order + l2_norm + ldist_reg
        return loss, loss_order, ldist_reg, l2_norm

    def loss_dist(self, scores_pred, z):
        mask = torch.where(z > self.delta_z_max, 0, 1)
        scores_diff = (scores_pred[:, 1:] - scores_pred[:, :-1]) * mask
        loss = self.l1loss(scores_diff[:, 1:], scores_diff[:, :-1])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def compute_slice_score_matrix(self, dataset, inference_device="cuda"):
        with torch.no_grad():
            self.eval()
            self.to(inference_device)
            slice_score_matrix = np.full(dataset.landmark_matrix.shape, np.nan)

            for i, slices, defined_landmarks in zip(
                np.arange(0, slice_score_matrix.shape[0]),
                dataset.landmark_slices_per_volume,
                dataset.defined_landmarks_per_volume,
            ):
                scores = self(
                    torch.tensor(slices[:, np.newaxis, :, :]).to(inference_device)
                )
                slice_score_matrix[i, defined_landmarks] = (
                    scores[:, 0].cpu().detach().numpy()
                )
        return slice_score_matrix

    def predict_tensor(self, tensor, n_splits=200, inference_device="cuda"):
        scores = []
        n = tensor.shape[0]
        slice_splits = list(np.arange(0, n, n_splits))
        slice_splits.append(n)

        with torch.no_grad():
            self.eval()
            self.to(inference_device)
            for i in range(len(slice_splits) - 1):
                min_index = slice_splits[i]
                max_index = slice_splits[i + 1]
                score = self(tensor[min_index:max_index, :, :, :].to(inference_device))
                scores += [s.item() for s in score]

        scores = np.array(scores)
        return scores

    def predict_npy_array(self, x, n_splits=200, inference_device="cuda"):
        x_tensor = torch.tensor(x[:, np.newaxis, :, :]).to(inference_device)
        scores = self.predict_tensor(
            x_tensor, inference_device=inference_device, n_splits=n_splits
        )
        return scores
