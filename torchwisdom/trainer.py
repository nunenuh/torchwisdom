import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torchwisdom.utils.data.collector import DatasetCollector
from torchwisdom.statemgr.state import *
from torch.optim import Optimizer
from torchwisdom.callback import *
from torchwisdom.metrics.callback import LossCallback, AccuracyCallback
from torchwisdom.statemgr.callback import StateManagerCallback
from torchwisdom.pbar import ProgressBarCallback
from torchwisdom.optim.wrapper import *
from typing import *


__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, data: DatasetCollector, model: Module, criterion: Module, metrics: List,
                 optimizer: Optimizer = None, callbacks: List = None, device='cpu'):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.device = device
        self.callbacks = callbacks
        self.optwr: OptimizerWrapper = None

        # self._build_state_manager()
        # self._build_callback_handler()

    def _build_optimizer(self, lr, **kwargs):
        self.optwr = OptimizerWrapper(self.model)
        self.optimizer = self.optwr.create(lr, self.optimizer, **kwargs)

    def _build_callback_handler(self):
        self.cb_handler: CallbackHandler = CallbackHandler(trainer=self)
        default_metrics: List[Callback] = [LossCallback()]
        default_callback: List[Callback] = [StateManagerCallback(), ProgressBarCallback()]
        clbks: List[Callback] = default_callback + default_metrics

        if self.metrics is not None:
            if type(self.metrics) is list:
                clbks = clbks + self.metrics
            elif isinstance(self.metrics, Callback):
                clbks.append(self.metrics)

        if self.callbacks is not None:
            if type(self.callbacks) is list:
                clbks = clbks + self.callbacks
            elif isinstance(self.callbacks, Callback):
                clbks.append(self.callbacks)

        self.cb_handler.add(clbks)
        self.cb_handler.rearrange_callback()

    def _build_state_manager(self):
        self.state_manager: StateManager = StateManager()

    def train(self, epoch, mbar):
        return NotImplementedError()

    def validate(self, epoch, mbar):
        return NotImplementedError()

    def fit(self, epoch_num):
        return NotImplementedError()

    def predict(self, images: torch.Tensor, use_topk: bool = False, topk: int = 5):
        pass

    def freeze(self, start: int, to: int = None):
        return NotImplementedError()

    def unfreeze(self):
        return NotImplementedError()

    def save(self):
        return NotImplementedError()

    def load(self):
        return NotImplementedError()

    def export(self):
        return NotImplementedError()



def predict_batch(model: nn.Module, features: torch.Tensor, use_topk=False, topk=5):
    model.eval()
    with torch.no_grad():
        output = model.forward(features)
        if use_topk:
            output = F.log_softmax(output, dim=1)
            ps = torch.exp(output)
            result = ps.topk(topk, dim=1, largest=True, sorted=True)
            return result
        else:
            return output


def predict_single(model: nn.Module, feature: torch.Tensor, use_topk=False, topk=5):
    feature = feature.unsqueeze(dim=0)
    with torch.no_grad():
        output = model.forward(feature)
        if use_topk:
            output = F.log_softmax(output, dim=1)
            ps = torch.exp(output)
            result = ps.topk(topk, dim=1, largest=True, sorted=True)
            result = result[0].squeeze(), result[1].squeeze()
            return result
        else:
            return output.squeeze()
