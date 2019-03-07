import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(input, target, smooth=1.):
    input_flat = input.view(-1)
    target_flat = target.view(-1)

    intersection = (input_flat * target_flat).sum()
    loss = (2 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
    return 1 - loss





