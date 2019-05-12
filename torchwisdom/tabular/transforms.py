import random
import PIL
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


__all__ = ['NumpyToTensor']


class NumpyToTensor(object):
    def __init__(self):
        super(NumpyToTensor, self).__init__()

    def __call__(self, np_array):
        return torch.from_numpy(np_array)

class ToFloatTensor(object):
    def __init__(self):
        super(ToFloatTensor, self).__init__()
    def __call__(self, x: torch.Tensor):
        return x.float()

class ToLongTensor(object):
    def __init__(self):
        super(ToLongTensor, self).__init__()
    def __call__(self, x: torch.Tensor):
        return x.long()

