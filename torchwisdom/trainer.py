import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer
from torchwisdom.utils.data.collector import DatasetCollector

class Trainer(object):
    def __init__(self, data: DatasetCollector, model: Module,
                 optimizer: Optimizer, criterion: Module, metrics:list,
                 callbacks=None, device='cpu'):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.device = device
        self.callbacks=callbacks

    def train(self):
        return NotImplementedError()

    def valid(self):
        return NotImplementedError()

    def fit(self):
        return NotImplementedError()

    def predict(self):
        return NotImplementedError()
