import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn import functional as N


# inspiration from
# https://github.com/pytorch/pytorch/issues/1249
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/core.py
# and other source
class DiceLoss(nn.Module):
    def __init__(self, sigmoid_norm: bool = False, softmax_norm: bool = False):
        super(DiceLoss, self).__init__()
        self.sigmoid_norm = sigmoid_norm
        self.softmax_norm = softmax_norm
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.sigmoid_norm:
            y_pred = torch.sigmoid(y_pred)
        if self.softmax_norm:
            y_pred = self.softmax(y_pred)
        return N.dice_coeff(y_pred, y_true)


class SoftDiceLoss(nn.Module):
    def __init__(self, sigmoid_norm: bool = False, softmax_norm: bool = False):
        super(SoftDiceLoss, self).__init__()
        self.sigmoid_norm = sigmoid_norm
        self.softmax_norm = softmax_norm
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pred: torch.Tensor, y_true) -> torch.Tensor:
        if self.sigmoid_norm:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = self.softmax(y_pred)
        return 1 - N.dice_coeff(y_pred, y_true)


# inspiration from
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/core.py
# and other source
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        bce_loss = self.bce_loss(input_flat, target_flat).double()
        dice_loss = self.dice_loss(input, target)
        return bce_loss + dice_loss


# inspired from https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b#file-contrastive-loss-py
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, y_predz, y_predj, y_true):
        dw = F.pairwise_distance(y_predz, y_predj)
        loss_similar = torch.mean((1 - y_true) * torch.pow(dw, 2))
        loss_dissimilar = torch.mean(y_true * torch.pow(torch.clamp(self.margin - dw, min=0.0), 2))
        loss_contrastive = loss_similar + loss_dissimilar
        #
        # euclidean_distance = F.pairwise_distance(y_predz, y_predj)
        # loss_contrastive = torch.mean((1 - y_true) * torch.pow(euclidean_distance, 2) +
        #                               (y_true) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
