from torchwisdom.utils.data.collector import DatasetCollector
from torch.nn.modules.module import Module
from torch.optim import Optimizer

class Trainer(object):
    def __init__(self, data:DatasetCollector, model:Module,
                 optimizer:Optimizer, lossfn:Module, metrics:list):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.metrics = metrics

    def train(self):
        return NotImplementedError()

    def valid(self):
        return NotImplementedError()

    def fit(self):
        return NotImplementedError()

    def predict(self):
        return NotImplementedError()








