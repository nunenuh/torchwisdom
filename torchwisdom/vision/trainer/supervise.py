import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ...core.callback import *
from ...core.data import DataCapsule
from ..data.capsule import *
from ...core.trainer import ClassifierTrainer
from ...core.exporter import *
from ...vision.predictor import ConvClassifierPredictor

__all__ = ['ConvClassifierTrainer']


class ConvClassifierTrainer(ClassifierTrainer):
    def __init__(self, data: ImageClassfierDataCapsule, model: nn.Module,
                 criterion: nn.Module = None, optimizer: Optimizer = None,
                 metrics: Collection[Callback] = None, callbacks: Collection[Callback] = None):
        super(ConvClassifierTrainer, self).__init__(data=data, model=model,
                                                    criterion=criterion, optimizer=optimizer,
                                                    metrics=metrics, callbacks=callbacks)

        self.predictor: ConvClassifierPredictor = None
        self._set_device()


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
        # print(feature.shape)
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

    def _init_predictor(self):
        self.exporter = VisionExporter(self)
        self.exporter.init_state()
        self.predictor: ConvClassifierPredictor = ConvClassifierPredictor(self.exporter.state)

    def predict(self, feature: Union[AnyStr, torch.Tensor], use_topk=False, kval=5, transform=None):
        if transform: self.predictor.transform = transform
        self._init_predictor()
        result = self.predictor.predict(feature, use_topk=use_topk, kval=kval)
        return result

    def export(self, path: str):
        self.exporter = VisionExporter(self)
        self.exporter.export(path)
