import random
import PIL
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


__all__ = ['NumpyToTensor', 'NumpyToPIL']


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


class TensorToNumpy(object):
    def __init__(self):
        super(TensorToNumpy, self).__init__()

    def __call__(self, tensor):
        # check if tensor is cpu
        # check if tensor is gpu
        # check if tensor is grad
        # use all that if to extract tensor from it
        return NotImplementedError()



if __name__ == '__main__':
    pass