import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import *
from .utils import DatasetCollector

__all__ = []


class _Predictor(object):
    def __init__(self, model: nn.Module, data: DatasetCollector, transform=None):
        self.device = 'cpu'
        self.transform: transforms.Compose = transform
        self.model: nn.Module = model
        self.data: DatasetCollector = data

    def _pre_check(self, data):
        return NotImplementedError()

    def _pre_predict(self, data):
        return NotImplementedError()

    def _predict(self, data):
        return NotImplementedError()

    def _post_check(self, data):
        return NotImplementedError

    def _post_predict(self, data):
        return NotImplementedError()

    def predict(self, data):
        return NotImplementedError()


class VisionSupervisePredictor(_Predictor):
    def __init__(self,  model: nn.Module, data: DatasetCollector, transform=None):
        super(VisionSupervisePredictor, self).__init__(model, data, transform)


class VisionUnsupervisePredictor(_Predictor):
    def __init__(self, model: nn.Module, data: DatasetCollector, transform=None):
        super(VisionUnsupervisePredictor, self).__init__(model, data, transform)


class TabularSupervisedPredictor(_Predictor):
    def __init__(self):
        super(TabularSupervisedPredictor, self).__init__()
        
        
class TabularUnsupervisedPredictor(_Predictor):
    def __init__(self):
        super(TabularUnsupervisedPredictor, self).__init__()
        

class TextSupervisedPredictor(_Predictor):
    def __init__(self):
        super(TextSupervisedPredictor, self).__init__()
        

class TextUnsupervisedPredictor(_Predictor):
    def __init__(self):
        super(TextUnsupervisedPredictor, self).__init__()
        

class AudioSupervisedPredictor(_Predictor):
    def __init__(self):
        super(AudioSupervisedPredictor, self).__init__()


class AudioUnsupervisedPredictor(_Predictor):
    def __init__(self):
        super(AudioUnsupervisedPredictor, self).__init__()


