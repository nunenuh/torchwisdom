import torch
import torch.nn as nn
from fastprogress import master_bar, progress_bar
from torch.nn.modules.module import Module
from torch.optim import Optimizer

from torchwisdom.core.optim.wrapper import OptimizerWrapper
from torchwisdom.core.utils.data import DatasetCollector
from .base import Trainer
from .helpers import *


__all__ = ['SemiSuperviseTrainer']


class SemiSuperviseTrainer(Trainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(SemiSuperviseTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.bunch = self.data.bunch()

    def compile(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def _build_callback_handler(self):
        self.handler = build_default_callback_handler(self)

    def _build_callback_handler_resume(self):
        self.handler = build_resume_callback_handler(self)

    def resumeable(self, val: bool):
        self.log_state = val

    def _build_state_manager(self):
        self.state_manager: StateManager = StateManager()

    def _build_optimizer(self, lr, **kwargs):
        self.optwr = OptimizerWrapper(self.model)
        self.optimizer = self.optwr.create(lr, self.optimizer, **kwargs)

    def _backward(self, loss: nn.Module):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _train_forward(self, feature, target):
        self.handler.on_train_forward_begin(feature=feature, target=target)
        pred = self._forward(feature)
        loss = self._loss_fn(pred, target)
        self.handler.on_train_forward_end(loss=loss, y_pred=pred, y_true=target)
        return loss

    def _train_backward(self, loss):
        self.handler.on_train_backward_begin()
        self._backward(loss)
        self.handler.on_train_backward_end()

    def _train(self, epoch, mbar: master_bar):
        self.handler.on_train_begin( master_bar=mbar)
        self.model.train()
        train_loader = self.bunch['train']
        trainbar = progress_bar(train_loader, parent=mbar)
        for idx, (feature, target) in enumerate(trainbar):
            self.handler.on_train_batch_begin(batch_curr=idx, master_bar=mbar)
            feature = feature.to(device=self.device)
            target = target.to(device=self.device)

            loss = self._train_forward(feature, target)
            self._train_backward(loss)

            self.handler.on_train_batch_end(master_bar=mbar)
        self.handler.on_train_end()

    def _validate_forward(self, feature, target):
        self.handler.on_validate_forward_begin(feature=feature, target=target)
        pred = self._forward(feature)
        loss = self._loss_fn(pred, target)
        self.handler.on_validate_forward_end(loss=loss, y_pred=pred, y_true=target)
        return loss

    def _validate(self, epoch, mbar: master_bar):
        self.handler.on_validate_begin(master_bar=mbar)
        self.model.eval()
        valid_loader = self.bunch['valid']
        progbar = progress_bar(valid_loader, parent=mbar)
        with torch.no_grad():
            for idx, (feature, target) in enumerate(progbar):
                self.handler.on_validate_batch_begin(batch_curr=idx, master_bar=mbar)
                feature = feature.to(device=self.device)
                target = target.to(device=self.device)

                self._validate_forward(feature, target)
                self.handler.on_validate_batch_end( master_bar=mbar)
        self.handler.on_validate_end(epoch=epoch,  master_bar=mbar)

    def fit(self, epoch_num, lr=0.01, wd=0, verbose=False, callbacks=None, **kwargs):
        self._build_optimizer(lr, weight_decay=wd, **kwargs)
        self._build_state_manager()
        self._build_callback_handler()  # CallbackHandler need to be the last to build

        mbar = master_bar(range(epoch_num))
        self.handler.on_fit_begin(epoch_num=epoch_num, master_bar=mbar)
        for epoch in mbar:
            self.handler.on_epoch_begin(epoch=epoch, master_bar=mbar)
            epoch = epoch + 1
            self._train(epoch, mbar)
            self._validate(epoch, mbar)
            self.handler.on_epoch_end(epoch=epoch, master_bar=mbar)
        self.handler.on_fit_end(epoch=epoch, master_bar=mbar)

    def _resume_load(self, id, from_last):
        self._build_state_manager()
        if id is not None:
            self.state_manager.load(id)
        if from_last:
            self.state_manager.load_last()
        self.optimizer = self.state_manager.state.get('optimizer').get("object")
        self.model = self.state_manager.state.get('model').get("object")
        self.criterion = self.state_manager.state.get('criterion')

    def resume(self, id: str = None, from_last: bool = True, **kwargs):
        self._resume_load(id, from_last)

        trainer_state: Dict = self.state_manager.state.get('trainer')
        lr = trainer_state.get("lr")
        epoch_curr = trainer_state.get("epoch").get("curr")
        epoch_num = trainer_state.get("epoch").get("num")
        self._build_optimizer(lr)
        self._build_callback_handler_resume()# CallbackHandler need to be the last to build

        mbar = master_bar(range(epoch_curr-1, epoch_num))
        self.handler.on_resume_begin(epoch_num=epoch_num, master_bar=mbar)
        for epoch in mbar:
            self.handler.on_epoch_begin(epoch=epoch, master_bar=mbar)
            epoch = epoch + 1
            self._train(epoch, mbar)
            self._validate(epoch, mbar)
            self.handler.on_epoch_end(epoch=epoch, master_bar=mbar)
        self.handler.on_resume_end(epoch=epoch, master_bar=mbar)
