from pathlib import Path
from .datasets import CSVDataset
from ..core.data import DataCapsule
from torch.utils.data import Dataset
from typing import *


class TabularDataCapsule(DataCapsule):
    def __init__(self, trainset: CSVDataset, validset: CSVDataset, testset: CSVDataset = None,
                 batch_size=64, shuffle=True, num_workers=2, data_type='categorical'):
        super(TabularDataCapsule, self).__init__(trainset, validset, testset, batch_size, shuffle, num_workers)
        self.trainset: CSVDataset = trainset
        self.validset: CSVDataset = validset
        self.testset: CSVDataset = testset
        self.valid_size = trainset.valid_size
        self.target_dtype = trainset.target_dtype
        self.drop_columns = trainset.drop_columns
        self.use_normalization = trainset.use_normalization
        self.normalization_mode = trainset.normalization_mode
        self.trn_feat_transform = trainset.transform
        self.trn_targ_transform = trainset.target_transform
        self.val_feat_transform = validset.transform
        self.val_targ_transform = validset.target_transform

        self.transform = validset.transform
        self.target_columns = trainset.target_columns
        self.feature_columns = trainset.feature_columns
        self.classes = trainset.classes
        self.class_idx = None
        self.case_type = 'tabular'
        self.data_type = 'categorical'


def tabular_data(csv_path: str, target_columns: List, target_dtype: str = 'categorical',
                 batch_size: int = 32, shuffle: bool = True, num_workers: int = 2, valid_size: float = 0.2,
                 feature_columns: List = None, drop_columns: List = [],
                 use_normalization: bool = False, normalization_mode: str = 'minmax',
                 trn_feat_transform: object = None, trn_targ_transform: object = None,
                 val_feat_transform: object = None, val_targ_transform: object = None):
    trainset = CSVDataset(csv_path, mode='train', target_dtype=target_dtype,
                          target_columns=target_columns, feature_columns=feature_columns,
                          drop_columns=drop_columns, valid_size=valid_size,
                          transform=trn_feat_transform, target_transform=trn_targ_transform,
                          use_normalization=use_normalization, normalization_mode=normalization_mode)

    validset = CSVDataset(csv_path, mode='valid', target_dtype=target_dtype,
                          target_columns=target_columns, feature_columns=feature_columns,
                          drop_columns=drop_columns, valid_size=valid_size,
                          transform=val_feat_transform, target_transform=val_targ_transform,
                          use_normalization=use_normalization, normalization_mode=normalization_mode)

    data_capsule = TabularDataCapsule(trainset=trainset, validset=validset,
                                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return data_capsule


def categorical_data(root, target_columns, batch_size=64, valid_size=0.2, **kwargs) -> DataCapsule:
    return tabular_data(root, target_columns=target_columns,
                        target_dtype='categorical', batch_size=batch_size,
                        valid_size=valid_size, **kwargs)


def regression_data(root, target_columns, batch_size=64, valid_size=0.2, **kwargs) -> DataCapsule:
    return tabular_data(root, target_columns=target_columns,
                        target_dtype='continues', batch_size=batch_size,
                        valid_size=valid_size, **kwargs)
