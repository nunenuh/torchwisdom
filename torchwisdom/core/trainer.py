import torch
from torch.nn.modules.module import Module
from torchwisdom.core.utils.data import DatasetCollector
from torchwisdom.core.statemgr.state import *
from torch.optim import Optimizer
from torchwisdom.core.callback import *
from torchwisdom.core.metrics.callback import LossCallback
from torchwisdom.core.statemgr.callback import StateManagerCallback
from torchwisdom.core.progress import ProgressBarCallback
from torchwisdom.core.nn.callback import *
from typing import *
from collections import OrderedDict


__all__ = ['Trainer', 'SuperviseTrainer', 'SemiSuperviseTrainer',
           'ClassifierTrainer', 'RegressorTrainer',
           'AutoEncoderTrainer', 'SiameseTrainer', 'GANTrainer']


class Trainer(object):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):

        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.device = "cpu"
        self.callbacks = callbacks
        self.optwr: OptimizerWrapper = None
        self.log_state = False
        self._set_device()

    def compile(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def _set_device(self):
        self.model = self.model.to(device=self.device)

    def resumeable(self, val: bool):
        self.log_state = val

    def _build_optimizer(self, lr, **kwargs):
        self.optwr = OptimizerWrapper(self.model)
        self.optimizer = self.optwr.create(lr, self.optimizer, **kwargs)

    def _build_callback_handler(self):
        self.cb_handler: CallbackHandler = CallbackHandler(trainer=self)
        default_metrics: List[Callback] = [LossCallback()]
        default_callback: List[Callback] = [StateManagerCallback(), ProgressBarCallback(),
                                            ModelCallback(), OptimizerCallback()]
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

    def _build_callback_handler_resume(self):
        self.cb_handler: CallbackHandler = CallbackHandler(trainer=self)
        callbacks_odict: OrderedDict = self.state_manager.state.get("callbacks")
        cbs = []
        for name, callback in callbacks_odict.items():
            cbs.append(callback)
        self.cb_handler.add(cbs)

    def _build_state_manager(self):
        self.state_manager: StateManager = StateManager()

    def train(self, epoch, mbar):
        return NotImplementedError()

    def validate(self, epoch, mbar):
        return NotImplementedError()

    def fit(self, epoch_num, lr, wd):
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

    def resume(self, from_last: bool = True, id: str = None, **kwargs):
        return NotImplementedError()


class SuperviseTrainer(Trainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(SuperviseTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)


class SemiSuperviseTrainer(Trainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(SemiSuperviseTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)


class ClassifierTrainer(SuperviseTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(ClassifierTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)


class RegressorTrainer(SuperviseTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(RegressorTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)


class AutoEncoderTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(AutoEncoderTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)


class SiameseTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(SiameseTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)


class GANTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(GANTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)
