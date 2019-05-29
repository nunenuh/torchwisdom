import torch
from typing import *
import torchvision.transforms as transforms
from ..transforms import pair as pair_transforms

__all__ = ['get_transforms', 'get_pair_transforms', 'get_imagenet_mean_std',
           'imagenet_normalize', 'imagenet_train_transform', 'imagenet_valid_transform',
           'std_train_pair_transform', 'pair_imagenet_normalize', 'std_valid_pair_transform']


def get_transforms(resize: Tuple = None, resized_crop: Tuple = None,
                   center_crop: Tuple = None, random_crop: Tuple = None,
                   grayscale: bool = False, rotate: float = None,
                   hflip: bool = False, vflip: bool = False,
                   normalize: bool = False, norm_mean: Tuple = None, norm_std: Tuple = None):
    tlist = []
    if resize: tlist.append(transforms.Resize(resize))
    if grayscale: tlist.append(transforms.Grayscale())
    if resized_crop: tlist.append(transforms.RandomResizedCrop(resized_crop))
    if center_crop: tlist.append(transforms.CenterCrop(center_crop))
    if random_crop: tlist.append(transforms.RandomCrop(random_crop))
    if rotate: tlist.append(transforms.RandomRotation(rotate))
    if hflip: tlist.append(transforms.RandomHorizontalFlip())
    if vflip: tlist.append(transforms.RandomVerticalFlip())

    tlist.append(transforms.ToTensor())
    if normalize:
        if norm_mean is None or norm_std is None:
            raise ValueError("norm_mean or norm_std cannot be none when normalize is True!"
                             " Both of them need to be initialized!")
        else:
            tlist.append(transforms.Normalize(norm_mean, norm_std))
    tmft = transforms.Compose(tlist)
    return tmft


def get_pair_transforms(resize: Tuple = None, resized_crop: Tuple = None,
                        center_crop: Tuple = None, random_crop: Tuple = None,
                        grayscale: bool = False, rotate: float = None,
                        hflip: bool = False, vflip: bool = False,
                        normalize: bool = False, norm_mean: Tuple = None, norm_std: Tuple = None):
    tlist = []
    if resize: tlist.append(pair_transforms.PairResize(resize))
    if grayscale: tlist.append(pair_transforms.PairGrayscale())
    if resized_crop: tlist.append(pair_transforms.PairRandomResizedCrop(resized_crop))
    if center_crop: tlist.append(pair_transforms.PairCenterCrop(center_crop))
    if random_crop: tlist.append(pair_transforms.PairRandomCrop(random_crop))
    if rotate: tlist.append(pair_transforms.PairRandomRotation(rotate))
    if hflip: tlist.append(pair_transforms.PairRandomHorizontalFlip())
    if vflip: tlist.append(pair_transforms.PairRandomVerticalFlip())

    tlist.append(pair_transforms.PairToTensor())
    if normalize:
        if norm_mean is None or norm_std is None:
            raise ValueError("norm_mean or norm_std cannot be none when normalize is True!"
                             " Both of them need to be initialized!")
        else:
            tlist.append(pair_transforms.PairNormalize(norm_mean, norm_std))
    tmft = pair_transforms.PairCompose(tlist)
    return tmft


def get_imagenet_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std


def imagenet_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def pair_imagenet_normalize():
    return pair_transforms.PairNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def imagenet_train_transform(image_size: Tuple = (224, 224), resize_crop: Tuple = (224, 224),
                             normalize=True, grayscale=False,
                             hflip=False, vlip=False, rotate=None, ):
    mean, std = get_imagenet_mean_std()
    tmft = get_transforms(resize=image_size, resize_crop=resize_crop, grayscale=grayscale,
                          hflip=hflip, vflip=vlip, rotate=rotate,
                          normalize=normalize, norm_mean=mean, norm_std=std)
    return tmft


def imagenet_valid_transform(image_size: Tuple = (224, 224), normalize: bool = True, grayscale: bool = False):
    mean, std = get_imagenet_mean_std()
    tmft = get_transforms(resize=image_size, center_crop=image_size, grayscale=grayscale,
                          normalize=normalize, norm_mean=mean, norm_std=std)
    return tmft


def std_train_pair_transform(image_size: Tuple = (224, 224), resized_crop: Tuple = (224, 224),
                             normalize=True, grayscale=False,
                             hflip=False, vlip=False, rotate=None):
    mean, std = get_imagenet_mean_std()
    tmft = get_pair_transforms(resize=image_size, resized_crop=resized_crop, grayscale=grayscale,
                               hflip=hflip, vflip=vlip, rotate=rotate,
                               normalize=normalize, norm_mean=mean, norm_std=std)

    return tmft


def std_valid_pair_transform(image_size: Tuple = (224, 224), normalize: bool = True, grayscale: bool = False):
    tmft = pair_transforms.PairCompose([
        pair_transforms.PairResize(image_size),
        pair_transforms.PairCenterCrop(image_size),
    ])
    if grayscale: tmft.transforms.append(pair_transforms.PairGrayscale())
    tmft.transforms.append(pair_transforms.PairToTensor())

    if normalize:
        tmft.transforms.append(pair_imagenet_normalize())

    return tmft


if __name__ == '__main__':
    tmft = get_transforms()
    print(tmft)
