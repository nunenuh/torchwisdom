from typing import *
import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer

from torchwisdom.core.trainer import SemiSuperviseTrainer
from torchwisdom.core.data import DataCapsule
from torchwisdom.vision.predictor import ConvAutoEncoderPredictor
from ...core.exporter import *

__all__ = ['AutoEncoderTrainer', 'SiamesePairTrainer', 'GANTrainer']


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


class SiamesePairTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DataCapsule, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(SiamesePairTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)

        self.predictor: ConvAutoEncoderPredictor = None
        self._set_device()

        # self._build_predictor()

    def _init_default(self):
        self.optimizer = 'adam'

    def compile(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion

    def _build_predictor(self):
        self.exporter = VisionExporter(self)
        self.exporter.init_state()
        self.predictor: ConvAutoEncoderPredictor = ConvAutoEncoderPredictor(self.exporter.state)

    def _data_loss_check_clean(self, pred, target):
        name = self.criterion.__class__.__name__
        target = target.float()
        # if name == 'BCELoss' or name == 'BCEWithLogitsLoss':
        #     pred = pred.contiguous().view(-1)
        #     target = target.contiguous().view(-1)
        # if name == 'MSELoss':
        #     pred = pred.view(-1)
        #     target = target.view(-1)
        #     pred = pred.float()
        #     target = target.float()
        return pred, target

    def data_to_device(self, feature, target):
        im1 = feature[0].to(self.device)
        im2 = feature[1].to(self.device)
        feature = (im1, im2)
        target = target.to(self.device)
        return feature, target

    def _loss_fn(self, pred, target):
        pred, target = self._data_loss_check_clean(pred, target)
        loss = self.criterion(pred[0], pred[1], target)
        return loss

    def _forward(self, feature):
        pred1, pred2 = self.model(feature[0], feature[1])
        return pred1, pred2

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


class GANTrainer(SemiSuperviseTrainer):
    def __init__(self, data: DataCapsule, model: Module,
                 criterion: Module = None, optimizer: Optimizer = None,
                 metrics: List = [], callbacks: List = None):
        super(GANTrainer, self).__init__(data, model, criterion, optimizer, metrics, callbacks)
