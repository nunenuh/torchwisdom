import torch
import torch.nn as nn
from .helpers import *
from ..utils.data import DatasetCollector
import torch.optim as optim
from ..optim.wrapper import *

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, data: DatasetCollector, model: nn.Module,
                 criterion: nn.Module = None, optimizer: optim.Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.device = "cpu"
        self.callbacks = callbacks
        self.handler = None
        self.optwr: OptimizerWrapper = None
        self.log_state = False
        self.predictor = None
        self._set_device()

    def compile(self, optimizer, criterion): return NotImplementedError()
    def resumeable(self, val: bool): ...
    def _set_device(self): ...
    def _build_optimizer(self, lr, **kwargs): ...
    def _build_predictor(self): ...
    def _build_state_manager(self): ...
    def _build_callback_handler(self): ...
    def _build_callback_handler_resume(self): ...
    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor): return NotImplementedError()
    def _forward(self, feature: torch.Tensor): return NotImplementedError()
    def _backward(self, loss: nn.Module): return NotImplementedError()
    def _train(self, epoch, mbar): return NotImplementedError()
    def _validate(self, epoch, mbar): return NotImplementedError()
    def fit(self, epoch_num: int, lr, wd): return NotImplementedError()
    def resume(self, from_last: bool = True, id: str = None, **kwargs): return NotImplementedError()
    def predict(self, images: torch.Tensor, use_topk: bool = False, topk: int = 5): return NotImplementedError()
    def evaluate(self): return NotImplementedError()
    def freeze(self, start: int, to: int = None): return NotImplementedError()
    def unfreeze(self): return NotImplementedError()
    def save(self): return NotImplementedError()
    def load(self): return NotImplementedError()
    def export(self): return NotImplementedError()
