import torch
import pandas as pd
import numpy as np
from typing import *
from torch import Tensor
from ...core import utils

__all__ = ['normalization', 'standardization', 'dice_coeff', 'dice_loss']


def normalization(x: Any, xmin: Any, xmax: Any) -> Any:
    return (x - xmin) / (xmax - xmin)


def standardization(x: Any, mean: float, std: float) -> Any:
    return (x - mean) / std


# inspiration from
# https://github.com/pytorch/pytorch/issues/1249
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/utils.py
# and other source
def dice_coeff(y_pred: Tensor, y_true: Tensor, smooth: float = 1.) -> Tensor:
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    intersection = (y_pred * y_true).sum()
    cardinality = y_pred.sum() + y_true.sum()
    return (2. * intersection + smooth) / (cardinality + smooth)


def dice_loss(y_pred: Tensor, y_true: Tensor, smooth: float = 1.) -> Tensor:
    return 1 - dice_coeff(y_pred, y_true, smooth)
