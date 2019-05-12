import torch
from torch import Tensor
import torch.nn.functional as F
from typing import *
from ...core import utils


__all__ = ['accuracy', 'accuracy_topk', 'accuracy_threshold', 'error_rate', 'dice_coeff',
           'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error',
           'mean_squared_logarithmic_error', 'accuracy_regression']


# taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def accuracy_topk(y_pred: Tensor, y_true: Tensor, topk: tuple = (1,)) -> Tensor:
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
    acc = y_pred.long() == y_true.long()
    return acc.float().mean()


# idea and code  got from https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L30
def accuracy_threshold(y_pred: Tensor, y_true: Tensor, thresh: float = 0.5, sigmoid: bool = False) -> Tensor:
    if sigmoid: y_pred = F.sigmoid(y_pred)
    y_thresh = y_pred > thresh
    acc = y_thresh==y_true.byte()
    return acc.float().mean()


def error_rate(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return 1 - accuracy(y_pred, y_true)


def bce_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    y_pred, y_true = y_pred.float(), y_true.float()
    return F.binary_cross_entropy(y_pred, y_true)


def bce_accuracy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    acc = (1 - bce_loss(y_pred, y_true))*100
    if acc < 0: return torch.Tensor([0])
    return acc


def bce_loss_with_logits(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    y_pred, y_true = y_pred.float(), y_true.float()
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def bce_loss_with_logits_accuracy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    acc = (1 - bce_loss_with_logits(y_pred, y_true))*100
    if acc < 0: return torch.Tensor([0])
    return acc

# inspiration from
# https://github.com/pytorch/pytorch/issues/1249
# github.com/jeffwen/road_building_extraction/blob/master/src/utils/utils.py
# and other source
def dice_coeff(y_pred: Tensor, y_true: Tensor, smooth: float = 1.) -> Tensor:
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    intersection = (y_pred * y_true).sum()
    return 1 - (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


# got idea from fastai
def mean_absolute_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # if not utils.is_flatten_same_dim(y_pred, y_true): y_pred = utils.flatten_argmax(y_pred)
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    return torch.abs(y_true-y_pred).mean()


def mean_squared_error(y_pred: Tensor, y_true: Tensor,
                       size_average: Any = None, reduce: Any = None, reduction: str = 'mean') -> Tensor:
    if not utils.is_flatten_same_dim(y_pred, y_true):
        y_pred = utils.flatten_argmax(y_pred)
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    y_pred, y_true = y_pred.float(), y_true.float()
    return F.mse_loss(y_pred, y_true, size_average, reduce, reduction)


def root_mean_squared_error(y_pred: Tensor, y_true: Tensor,
                            size_average: Any = None, reduce: Any = None, reduction: str = 'mean') -> Tensor:
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    torch.sqrt(mean_squared_error(y_pred, y_true, size_average, reduce, reduction))


def mean_squared_logarithmic_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred, y_true = utils.flatten_check(y_pred, y_true)
    y_pred, y_true = torch.log(1+y_pred), torch.log(1+y_true)
    return F.mse_loss(y_pred, y_true)


# modified from
# https://jamesmccaffrey.wordpress.com/2018/11/26/an-efficient-accuracy-function-for-pytorch-regression/
def accuracy_regression(y_pred, y_true, thresh=0.5):
    thresh = 1.0 - thresh
    correct = torch.sum(torch.abs(y_pred - y_true) < torch.abs(thresh * y_true))
    acc = ((correct * 100.0) / len(y_true))
    return acc

