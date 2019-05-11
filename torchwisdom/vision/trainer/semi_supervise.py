from typing import *
import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer

from torchwisdom.core.trainer import SemiSuperviseTrainer
from torchwisdom.core.utils.data import DatasetCollector
from torchwisdom.vision.predictor import ConvClassifierPredictor, ConvAutoEncoderPredictor


__all__ = ['AutoEncoderTrainer', 'SiameseTrainer', 'GANTrainer']


class AutoEncoderTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(AutoEncoderTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.predictor: ConvAutoEncoderPredictor = None
        self._set_device()
        self._build_predictor()

    def compile(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def _build_predictor(self):
        self.predictor: ConvAutoEncoderPredictor = ConvAutoEncoderPredictor(self.model, self.data)

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

    def predict(self, images: torch.Tensor):
        return self.predictor.predict(images)


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
