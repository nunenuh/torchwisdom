import torch
import torch.nn as nn
import torch.nn.functional as F
from torchwisdom.callback import Callback
from typing import *

# taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMetrics(object):
    def __init__(self):
        self.restart()

    def restart(self):
        self.value = 0
        self.values = []
        self.mean = 0
        self.std = 0

    def update(self, val):
        self.value = val
        self.values.append(val)
        tensor = torch.Tensor(self.values)
        self.mean = tensor.mean().item()
        self.std = tensor.std().item()


class AverageMeterCallback(Callback):
    def __init__(self):
        self.avm = AverageMeter()

    def on_epoch_begin(self, **kwargs: Any):
        self.avm.reset()

    def on_batch_begin(self, val, batch_size):
        self.avm.update(val, batch_size)




class AccuracyTopK(object):
    def __init__(self, topk:tuple=(1,)):
        self.topk = topk

    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = y_true.size(0)

            _, pred = y_pred.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y_true.view(1, -1).expand_as(pred))

            res = []
            for k in self.topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


class Accuracy(object):
    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        bsize = y_pred.size(0)
        y_pred = y_pred.argmax(dim=-1).view(bsize, -1)
        y_true = y_true.view(bsize, -1)
        acc = (y_pred==y_true).float().mean()
        return acc


class AccuracyThreshold(object):
    def __init__(self, thresh=0.5, sigmoid=False):
        super(AccuracyThreshold, self).__init__()
        self.thresh = thresh
        self.sigmoid = sigmoid

    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        if self.sigmoid: y_pred = F.sigmoid(y_pred)
        y_thresh = y_pred > self.thresh
        acc = (y_thresh == y_true.byte()).float().mean()
        return acc

