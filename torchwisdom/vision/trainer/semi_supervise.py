from typing import *
import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer

from torchwisdom.core.trainer import SemiSuperviseTrainer
from torchwisdom.core.data import DataCapsule
from torchwisdom.vision.predictor import ConvAutoEncoderPredictor
from ...core.exporter import *



__all__ = ['AutoEncoderTrainer', 'SiameseTrainer', 'GANTrainer']


class AutoEncoderTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DataCapsule, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(AutoEncoderTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.predictor: ConvAutoEncoderPredictor = None
        self._set_device()
        # self._build_predictor()

    def compile(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def _build_predictor(self):
        self.exporter = VisionExporter(self)
        self.exporter.init_state()
        self.predictor: ConvAutoEncoderPredictor = ConvAutoEncoderPredictor(self.exporter.state)

    def _data_loss_check_clean(self, pred, target):
        name = self.criterion.__class__.__name__
        if name == 'BCELoss' or name == 'BCEWithLogitsLoss':
            pred = pred.contiguous().view(-1)
            target = target.contiguous().view(-1)
        if name == 'MSELoss':
            pred = pred.view(-1)
            target = target.view(-1)
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
        self.exporter = VisionExporter(self)
        self.exporter.init_state()
        self.predictor: ConvAutoEncoderPredictor = ConvAutoEncoderPredictor(self.exporter.state)

    def predict(self, data: torch.Tensor):
        self._init_predictor()
        return self.predictor.predict(data)

    def export(self, path: str):
        self.exporter = VisionExporter(self)
        self.exporter.export(path)


class SiameseTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DataCapsule, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(SiameseTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)


class GANTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DataCapsule, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(GANTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)
