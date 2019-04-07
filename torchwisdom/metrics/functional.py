import torch
from torch import Tensor
from typing import *
import torch.nn.functional as F
from torchwisdom import core


# taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def accuracy_topk(y_pred: Tensor, y_true: Tensor, topk: tuple=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = y_true.size(0)

        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# idea and code got from https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L23
def accuracy(y_pred: Tensor, y_true: Tensor):
    bsize = y_pred.size(0)
    y_pred = y_pred.argmax(dim=-1).view(bsize, -1)
    y_true = y_true.view(bsize, -1)
    acc = y_pred==y_true
    return acc.float().mean()


# idea and code  got from https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L30
def accuracy_threshold(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=False):
    if sigmoid: y_pred = F.sigmoid(y_pred)
    y_thresh = y_pred > thresh
    acc = y_thresh==y_true.byte()
    return acc.float().mean()

def error_rate(y_pred: Tensor, y_true:Tensor) -> Tensor:
    return 1 - accuracy(y_pred, y_true)

# inspiration from
# https://github.com/pytorch/pytorch/issues/1249
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/metrics.py
# and other source
def dice_coeff(y_pred, y_true, smooth=1.):
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)

    intersection = (y_pred_flat * y_true_flat).sum()
    return 1 - (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)


#got idea from fastai
def mean_absolute_error(y_pred: Tensor, y_true: Tensor)->Tensor:
    y_pred, y_true = core.flatten_check(y_pred, y_true)
    return torch.abs(y_true-y_pred).mean()

def mean_squared_error(y_pred: Tensor, y_true: Tensor, size_average=None, reduce=None, reduction='mean')->Tensor:
    y_pred, y_true = core.flatten_check(y_pred, y_true)
    return F.mse_loss(y_pred, y_true, size_average, reduce, reduction)

def root_mean_squared_error(y_pred: Tensor, y_true: Tensor, size_average=None, reduce=None, reduction='mean')->Tensor:
    y_pred, y_true = core.flatten_check(y_pred, y_true)
    torch.sqrt(mean_squared_error(y_pred, y_true, size_average, reduce, reduction))

def mean_squared_logarithmic_error(y_pred: Tensor, y_true: Tensor)->Tensor:
    y_pred, y_true = core.flatten_check(y_pred, y_true)
    y_pred, y_true = torch.log(1+y_pred), torch.log(1+y_true)
    return F.mse_loss(y_pred, y_true)
