import torch
from typing import *
import torchvision.transforms as transforms
from ..transforms import pair as pair_transforms

__all__ = ['imagenet_normalize', 'imagenet_train_transform', 'imagenet_valid_transform',
           'std_train_pair_transform', 'pair_imagenet_normalize', 'std_valid_pair_transform']


def imagenet_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def pair_imagenet_normalize():
    return pair_transforms.PairNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def imagenet_train_transform(image_size: Tuple = (224, 224), resize_crop: Tuple = (224, 224), normalize=True):
    tmft_compose = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(resize_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if normalize:
        tmft_compose.transforms.append(imagenet_normalize())
    return tmft_compose


def imagenet_valid_transform(image_size: Tuple = (224, 224), normalize=True):
    tmft_compose = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    if normalize:
        tmft_compose.transforms.append(imagenet_normalize())
    return tmft_compose


def std_train_pair_transform(image_size: Tuple = (224, 224), random_rotation: int = 10, normalize=True):
    tmft = pair_transforms.PairCompose([
        pair_transforms.PairResize(image_size),
        pair_transforms.PairCenterCrop(image_size),
        pair_transforms.PairRandomHorizontalFlip(),
        pair_transforms.PairRandomRotation(random_rotation),
        pair_transforms.PairToTensor(),
    ])
    if normalize:
        tmft.transforms.append(pair_imagenet_normalize())

    return tmft


def std_valid_pair_transform(image_size: Tuple = (224, 224), normalize=True):
    tmft = pair_transforms.PairCompose([
        pair_transforms.PairResize(image_size),
        pair_transforms.PairCenterCrop(image_size),
        pair_transforms.PairToTensor(),
    ])
    if normalize:
        tmft.transforms.append(pair_imagenet_normalize())

    return tmft
