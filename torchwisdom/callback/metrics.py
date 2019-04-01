from torchwisdom.callback.callback import Callback
from torchwisdom.metrics.metrics import *
import torchwisdom.metrics.functional as M
from typing import *


class AverageMetricsCallback(Callback):
    def __init__(self):
        self.metric = AverageMetrics()

    def on_epoch_begin(self, **kwargs: Any):
        self.metric.restart()

    def on_backward_end(self, val):
        self.metric.update(val)


class AccuracyCallback(AverageMetricsCallback):
    def __init__(self):
        super(AccuracyCallback, self).__init__()
        self.acc = Accuracy()

    def on_backward_end(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        self.metric.update(M.accuracy(y_pred, y_true))


class AccuracyTopKCallback(AverageMetricsCallback):
    def __init__(self, topk: tuple = (1,)):
        super(AccuracyTopKCallback, self).__init__()
        self.topk = topk

    def on_backward_end(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        self.metric.update(M.accuracy_topk(y_pred, y_true, self.topk))


class AccuracyThresholdCallback(AverageMetricsCallback):
    def __init__(self, threshold=0.5, sigmoid=False):
        self.threshold = threshold
        self.sigmoid = sigmoid

    def on_backward_end(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        self.metric.update(M.accuracy_threshold(y_pred, y_true, self.threshold, self.sigmoid))


