import torch
from PIL import Image
import PIL
import collections
import torchvision.transforms.functional as F
import torchvision
from torchvision import transforms

__all__ = ['PairCompose','PairResize']

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class PairCompose(transforms.Compose):
    def __init__(self, transforms):
        super(PairCompose, self).__init__(transforms)

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class PairResize(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
       super(PairResize, self).__init__(size, interpolation)

    def __call__(self, img1, img2):
        img1 = F.resize(img1, self.size, self.interpolation)
        img2 = F.resize(img2, self.size, self.interpolation)
        return img1, img2



