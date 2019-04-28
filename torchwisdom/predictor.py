import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import *


__all__ = ['ConvPredictor', ]


class _Predictor(object):
    def __init__(self, model: nn.Module, transform = None, ):
        self.device = 'cpu'
        self.transform: transforms.Compose = None
        self.model: nn.Module = None
        self.data: Union[str, torch.Tensor] = None

    def batch(self, data):
        return NotImplementedError()

    def single(self, data):
        return NotImplementedError()

    def _pre_predict(self, data) -> torch.Tensor:
        return NotImplementedError()

    def _predict(self, data) -> torch.Tensor:
        return NotImplementedError()

    def _post_predict(self, data) -> torch.Tensor:
        return NotImplementedError()


class _VisionSupervisePredictor(_Predictor):
    def __init__(self):
        super(_VisionSupervisePredictor, self).__init__()

    def batch(self, data):
        transformed_data = self.transform(data)
        transformed_data = transformed_data.to(self.device)

        self.model.to(self.device)
        self.model.eval()
        pred = self.model(transformed_data)
        return pred

    def single(self, data):
        pass

    def topk(self):
        pass


class _VisionUnsupervisePredictor(_Predictor):
    def __init__(self):
        super(_VisionUnsupervisePredictor, self).__init__()


class _TabularSupervisedPredictor(_Predictor):
    def __init__(self):
        super(_TabularSupervisedPredictor, self).__init__()
        
        
class _TabularUnsupervisedPredictor(_Predictor):
    def __init__(self):
        super(_TabularUnsupervisedPredictor, self).__init__()
        

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


class ConvPredictor(_VisionSupervisePredictor):
    def __init__(self):
        super(ConvPredictor, self).__init__()

