import torch
from torch import optim, nn
from torch.nn import Module
from torch.optim import Optimizer
from typing import *

from ..core.callback import AccuracyCallback, AccuracyRegressionCallback
from ..core.trainer.supervise import *
from ..core.data import DataCapsule
from ..tabular.predictor import TabularClassifierPredictor, TabularRegressorPredictor
from ..core.exporter import ExporterBuilder, TabularExporter


class TabularClassifierTrainer(ClassifierTrainer):
    def __init__(self, data: DataCapsule, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = None, callbacks: List = None):
        super(ClassifierTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.predictor: TabularClassifierPredictor = None
        self._set_device()

    def _init_predictor(self):
        self.exporter = TabularExporter(self)
        self.exporter.init_state()
        self.predictor: TabularClassifierPredictor = TabularClassifierPredictor(self.exporter.state)

    def _data_loss_check_clean(self, pred, target):
        name = self.criterion.__class__.__name__
        if name == 'BCELoss' or name == 'BCEWithLogitsLoss':
            pred = pred.float()
            target = target.unsqueeze(dim=1).float()
        if name == 'MSELoss':
            pred = pred.float()
            target = target.float()
        return pred, target

    def _loss_fn(self, pred, target):
        pred, target = self._data_loss_check_clean(pred, target)
        loss = self.criterion(pred, target)
        return loss

    def _forward(self, feature):
        pred = self.model(feature)
        return pred

    def freeze(self, last_from: int = -1, last_to: int = None):
        params = list(self.model.parameters())
        if last_to == None:
            for param in params[:last_from]:
                param.requires_grad = False

            for param in params[last_from:]:
                param.requires_grad = True
        else:
            for param in params[:last_to]:
                param.requires_grad = False

            for param in params[last_from:last_to]:
                param.requires_grad = True

    def unfreeze(self):
        params = self.model.parameters()
        for param in params:
            param.requires_grad = True

    def predict(self, *args: Any, feature=None, transform: object = None,
                use_topk: bool = False, kval: int = 2, **kwargs: Any):
        self._init_predictor()
        result = self.predictor.predict(feature, use_topk=use_topk, kval=kval, transform=transform, **kwargs)
        return result

    def export(self, path: str):
        self.exporter = TabularExporter(self)
        self.exporter.export(path)


class TabularRegressorTrainer(RegressorTrainer):
    def __init__(self, data: DataCapsule, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = None, callbacks: List = None):
        super(TabularRegressorTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.predictor: TabularRegressorPredictor = None
        self._set_device()
        # self._build_predictor()

    def _data_loss_check_clean(self, pred, target):
        name = self.criterion.__class__.__name__
        if name == 'BCELoss' or name == 'BCEWithLogitsLoss':
            pred = pred.float()
            target = target.unsqueeze(dim=1).float()
        if name == 'MSELoss':
            pred = pred.float()
            target = target.float()
        return pred, target

    def _loss_fn(self, pred, target):
        pred, target = self._data_loss_check_clean(pred, target)
        loss = self.criterion(pred, target)
        return loss

    def _forward(self, feature):
        pred = self.model(feature)
        return pred

    def _init_predictor(self):
        self.exporter = TabularExporter(self)
        self.exporter.init_state()
        self.predictor: TabularRegressorPredictor = TabularRegressorPredictor(self.exporter.state)

    def predict(self, feature, target=None, show_table=False, transform: object = None):
        self._init_predictor()
        result = self.predictor.predict(feature, target=target, show_table=show_table, transform=transform, )
        return result

    def export(self, path: str):
        self.exporter = TabularExporter(self)
        self.exporter.export(path)




def classifier_trainer(data, model, opt=optim.Adam, crit=nn.CrossEntropyLoss()):
    trainer = TabularClassifierTrainer(data, model)
    trainer.compile(optimizer=opt, criterion=crit)
    trainer.metrics = [AccuracyCallback()]
    return trainer


def regressor_trainer(data, model, opt=optim.Adam, crit=nn.MSELoss()):
    trainer = TabularRegressorTrainer(data, model)
    trainer.compile(optimizer=opt, criterion=crit)
    trainer.metrics = [AccuracyRegressionCallback(threshold=0.8)]
    return trainer
