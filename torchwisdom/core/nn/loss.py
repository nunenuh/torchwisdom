import torch
import torch.nn as nn
import torch.nn.functional as F

# inspiration from
# https://github.com/pytorch/pytorch/issues/1249
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/core.py
# and other source
def dice_coeff(input, target, smooth=1.):
    input_flat = input.view(-1)
    target_flat = target.view(-1)

    intersection = (input_flat * target_flat).sum()
    return (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        dcoeff = dice_coeff(input, target)
        return 1 - dcoeff

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

# taken from https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b#file-contrastive-loss-py
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive






