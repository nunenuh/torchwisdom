import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from ...core.data import DataCapsule
from ...vision import transforms as ptransforms
from .datasets import *
from ..transforms.helpers import *
from typing import *


class ImageClassfierDataCapsule(DataCapsule):
    def __init__(self, trainset: ImageFolder, validset: ImageFolder, testset: ImageFolder = None,
                 batch_size=64, shuffle=True, num_workers=2):
        super(ImageClassfierDataCapsule, self).__init__(trainset, validset, testset, batch_size, shuffle, num_workers)
        self.trainset: ImageFolder = trainset
        self.validset: ImageFolder = validset
        self.testset: ImageFolder = testset
        self.trn_feat_transform: transforms.Compose = trainset.transform
        self.trn_targ_transform: transforms.Compose = trainset.target_transform
        self.val_feat_transform: transforms.Compose = validset.transform
        self.val_targ_transform: transforms.Compose = validset.target_transform

        self.transform = validset.transform
        self.classes = trainset.classes
        self.class_idx = None
        self.case_type = 'vision'
        self.data_type = 'categorical'


def image_classifier_data(path: str, train_dir: str = 'train', valid_dir: str = 'valid',
                          batch_size: int = 32, shuffle: bool = True, num_workers: int = 2,
                          image_size: Tuple = (224, 224)):
    tmft_train = imagenet_train_transform(image_size=image_size, resize_crop=image_size)
    tmft_valid = imagenet_valid_transform(image_size)

    path = Path(path)
    train_path = str(path.joinpath(train_dir))
    trainset = ImageFolder(train_path, transform=tmft_train)

    valid_path = str(path.joinpath(valid_dir))
    validset = ImageFolder(valid_path, transform=tmft_valid)

    data_capsule = ImageClassfierDataCapsule(trainset, validset,
                                             batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers)
    return data_capsule


class AutoEncoderDataCapsule(DataCapsule):
    def __init__(self, trainset: AutoEncoderDataset, validset: AutoEncoderDataset, testset: AutoEncoderDataset = None,
                 batch_size=64, shuffle=True, num_workers=2):
        super(AutoEncoderDataCapsule, self).__init__(trainset, validset, testset, batch_size, shuffle, num_workers)
        self.trainset: AutoEncoderDataset = trainset
        self.validset: AutoEncoderDataset = validset
        self.testset: AutoEncoderDataset = testset

        self.trn_feat_transform: transforms.Compose = trainset.feature_transform
        self.trn_pair_transform: ptransforms.PairCompose = trainset.pair_transform
        self.trn_targ_transform: transforms.Compose = trainset.target_transform

        self.val_feat_transform: transforms.Compose = validset.feature_transform
        self.val_pair_transform: ptransforms.PairCompose = validset.pair_transform
        self.val_targ_transform: transforms.Compose = validset.target_transform

        self.transform = None
        self.case_type = 'vision'
        self.data_type = 'categorical'
        self.classes = None
        self.class_idx = None


def autoencoder_data(path: str, feature_dir: str, target_dir: str,
                     split_dataset: bool = False, valid_size: float = 0.2,
                     limit_size: Union[int, float] = 0, limit_type: Union[float, int] = int,
                     train_dir: str = 'train', valid_dir: str = 'valid',
                     batch_size: int = 32, shuffle: bool = True, num_workers: int = 2,
                     normalize=False, image_size: Tuple = (224, 224)):
    path = Path(path)
    if split_dataset:
        train_path = str(path)
        valid_path = str(path)
    else:
        train_path = str(path.joinpath(train_dir))
        valid_path = str(path.joinpath(valid_dir))

    trainset = AutoEncoderDataset(train_path, feature_dir, target_dir,
                                  split_dataset=split_dataset, valid_size=valid_size, mode='train',
                                  limit_size=limit_size, limit_type=limit_type,
                                  pair_transform=std_train_pair_transform(image_size, normalize))

    validset = AutoEncoderDataset(valid_path, feature_dir, target_dir,
                                  split_dataset=split_dataset, valid_size=valid_size, mode='valid',
                                  limit_size=limit_size, limit_type=limit_type,
                                  pair_transform=std_valid_pair_transform(image_size, normalize))

    data_capsule = AutoEncoderDataCapsule(trainset, validset,
                                          batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers)
    data_capsule.transform = imagenet_valid_transform(image_size, normalize)

    return data_capsule
