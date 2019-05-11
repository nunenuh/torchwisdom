import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from torchwisdom.core.callback import *
from torchwisdom.core.utils.data import DatasetCollector
from torchwisdom.vision.predictor import ConvClassifierPredictor
from torchwisdom.core.trainer import ClassifierTrainer

__all__ = ['ConvTrainer']


class ConvTrainer(ClassifierTrainer):
    def __init__(self, data: DatasetCollector, model: nn.Module,
                 criterion: nn.Module = None, optimizer: Optimizer = None,
                 metrics: Collection[Callback] = None, callbacks: Collection[Callback] = None):
        '''
        :param data:
        :param model:
        :param optimizer:
        :param criterion:
        :param metrics:
        :param device:

        '''
        super(ConvTrainer, self).__init__(data=data, model=model,
                                          criterion=criterion, optimizer=optimizer,
                                          metrics=metrics, callbacks=callbacks)


        self.predictor: ConvClassifierPredictor = None
        self._set_device()
        self._build_predictor()

    def _build_predictor(self):
        self.predictor: ConvClassifierPredictor = ConvClassifierPredictor(self.model, self.data)

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

    def predict(self, data: Union[AnyStr, torch.Tensor], use_topk=False, kval=5, transform=None):
        self.predictor.transform = transform
        result = self.predictor.predict(data, use_topk=use_topk, kval=kval)
        return result
