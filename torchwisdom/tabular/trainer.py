import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing import *
from torchwisdom.core.trainer.supervise import *
from torchwisdom.core.utils import DatasetCollector
from torchwisdom.tabular.predictor import TabularClassifierPredictor, TabularRegressorPredictor


class TabularClassifierTrainer(ClassifierTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = None, callbacks: List = None):
        super(ClassifierTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.predictor: TabularClassifierPredictor = None
        self._set_device()
        self._build_predictor()

    def _build_predictor(self):
        self.predictor: TabularClassifierPredictor = TabularClassifierPredictor(self.model, self.data)

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

    def predict(self, *args, **kwargs: Any):
        use_topk, kval, transform = kwargs.get("use_topk", False), kwargs.get("kval", 2), \
                                    kwargs.get("transform", None)
        kwargs['feature_columns'] = self.data.validset.feature_columns
        kwargs['target_columns'] = self.data.validset.target_columns

        if transform:
            self.predictor.transform = transform
        result = self.predictor.predict(*args, **kwargs)
        return result


class TabularRegressorTrainer(RegressorTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = None, callbacks: List = None):
        super(TabularRegressorTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.predictor: TabularRegressorPredictor = None
        self._set_device()
        self._build_predictor()

    def _build_predictor(self):
        self.predictor: TabularRegressorPredictor = TabularRegressorPredictor(self.model, self.data)

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

    def predict(self, *args, **kwargs: Any):
        use_topk, kval, transform = kwargs.get("use_topk", False), kwargs.get("kval", 2), \
                                    kwargs.get("transform", None)

        kwargs['feature_columns'] = self.data.validset.feature_columns
        kwargs['target_columns'] = self.data.validset.target_columns

        if transform:
            self.predictor.transform = transform
        result = self.predictor.predict(*args, **kwargs)

        return result



