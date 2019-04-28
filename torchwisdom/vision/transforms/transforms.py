import random
import PIL
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np

__all__ = []


class NumpyToTensor(object):
    def __init__(self):
        super(NumpyToTensor, self).__init__()

    def __call__(self, np_array):
        return torch.from_numpy(np_array)


class NumpyToPIL(object):
    def __init__(self):
        super(NumpyToPIL, self).__init__()

    def __call__(self, np_array):
        return Image.fromarray(np_array)




if __name__ == '__main__':
    pass