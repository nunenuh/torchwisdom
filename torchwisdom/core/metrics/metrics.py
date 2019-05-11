import torch
from ..metrics import functional as M

__all__ = ['AverageMetrics', 'AccuracyTopK', 'Accuracy', 'AccuracyThreshold',
           'DiceCoef', 'MAE', 'MSE', 'MSLE', 'RMSE']


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


class AccuracyTopK(object):
    def __init__(self, topk:tuple=(1,)):
        self.topk = topk

    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        return M.accuracy_topk(y_pred, y_true, self.topk)


class Accuracy(object):
    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
       return M.accuracy(y_pred, y_true)


class AccuracyThreshold(object):
    def __init__(self, thresh=0.5, sigmoid=False):
        super(AccuracyThreshold, self).__init__()
        self.thresh = thresh
        self.sigmoid = sigmoid

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return M.accuracy_threshold(y_pred, y_true, self.thresh, self.sigmoid)


class DiceCoef(object):
    def __init__(self, smooth=1.):
        self.smooth = smooth

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, smooth=None):
        if smooth: self.smooth = smooth
        return M.dice_coeff(y_pred, y_true, self.smooth)


class MAE(object):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return M.mean_absolute_error(y_pred, y_true)


class MSE(object):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                 size_average=None, reduce=None, reduction: str = 'mean'):
        return M.mean_squared_error(y_pred, y_true, size_average, reduce, reduction)


class MSLE(object):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return M.mean_squared_logarithmic_error(y_pred, y_true)


class RMSE(object):
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                 size_average=None, reduce=None, reduction: str = 'mean'):
        return M.root_mean_squared_error(y_pred, y_true, size_average, reduce, reduction)