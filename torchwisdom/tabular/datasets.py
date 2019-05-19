import random
from pathlib import Path
from typing import *

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import dataset
from ..tabular import transforms as tab_transforms

__all__ = ['CSVDataset']


class CSVDataset(dataset.Dataset):
    def __init__(self, file: str, target_columns: Union[str, List], feature_columns: list = None,
                 transform: Any = None, target_transform: Any = None,
                 target_dtype: str = 'categorical', drop_columns: List = [], valid_size: float = 0.0,
                 mode: str = 'train', use_normalization: bool = False, normalization_mode: str = 'minmax'):
        super(CSVDataset, self).__init__()
        self.file = Path(file)
        self.target_dtype = target_dtype
        self.drop_columns = drop_columns
        self.data_frame = self._file_check()
        self.train_frame = None
        self.valid_frame = None
        self.target_columns = target_columns
        self.feature_columns = feature_columns
        self.transform: Callable = transform
        self.target_transform: Callable = target_transform
        self.valid_size = valid_size
        self.mode = mode
        self.use_normalization = use_normalization
        self.normalization_mode = normalization_mode
        self._build_classes()
        self._dataset_split()
        self._build_column()
        self._build_feature_stats()
        self._build_default_transform()
        self._build_frame()

    def _file_check(self):
        if not self.file.exists():
            raise ValueError('File not found!, please check your filepath')
        elif not self.file.is_file():
            raise ValueError('Expected file as input in parameter file but, got other type')

        df = pd.read_csv(self.file)
        if len(self.drop_columns):
            df.drop(self.drop_columns, inplace=True, axis=1)
        return df

    def _build_feature_stats(self):
        self.feature_stats = {}
        frame = self.data_frame
        for col in self.feature_columns:
            self.feature_stats.update({col: {
                'min': frame[col].min(), 'max': frame[col].max(),
                'mean': frame[col].mean(), 'std': frame[col].std()
            }})

    def _build_classes(self):
        if self.target_dtype == 'categorical':
            frame = pd.get_dummies(self.data_frame[self.target_columns])
            classes_columns = list(frame.columns)
            self.classes = []
            for cls in classes_columns:
                self.classes.append(cls)
        else:
            self.classes = None

    def _get_index(self, classes_name):
        df2 = self.data_frame[self.data_frame[self.target_columns] == classes_name]
        df3 = df2[self.target_columns].dropna()
        return list(df3.index)

    def _dataset_split(self):
        random.seed(1261)

        if self.target_dtype == 'categorical':
            valid_index = []
            for cname in self.classes:
                list_index = self._get_index(cname)
                size = int(self.valid_size * len(list_index))
                valid_index += random.sample(list_index, size)
            train_index = list(set(list(self.data_frame.index)) - set(valid_index))
            train_index, valid_index = sorted(train_index), sorted(valid_index)
        else:
            list_index = list(self.data_frame.index)
            size = int(self.valid_size * len(list_index))
            valid_index = random.sample(list_index, size)
            train_index = list(set(list(self.data_frame.index)) - set(valid_index))
            train_index, valid_index = sorted(train_index), sorted(valid_index)

        self.valid_frame = self.data_frame.iloc[valid_index]
        self.valid_frame.index = pd.RangeIndex(len(self.valid_frame.index))

        self.train_frame = self.data_frame.iloc[train_index]
        self.train_frame.index = pd.RangeIndex(len(self.train_frame.index))

    def _proper_frame(self) -> pd.DataFrame:
        frame = self.train_frame
        if self.mode == 'valid':
            frame = self.valid_frame
        return frame

    def _build_column(self):
        frame = self._proper_frame()
        self.columns = list(frame.columns)
        columns = self.columns

        if type(self.target_columns) is str:
            self.target_columns = [self.target_columns]
        self._clean_check_column(columns, self.target_columns, mode='target')

        if not self.feature_columns:
            for col in self.target_columns:
                columns.remove(col)
            self.feature_columns = columns
        self._clean_check_column(columns, self.feature_columns, mode='feature')

    @staticmethod
    def _clean_check_column(all_columns: List, choosed_column: List, mode='target'):
        for coll in choosed_column:
            if coll not in all_columns:
                raise ValueError(f"{mode}_columns with value '{coll}' is not exist in csv columns!")

    def _build_frame(self):
        frame = self._proper_frame()
        if self.target_dtype == 'categorical':
            self.target_frame = pd.get_dummies(frame[self.target_columns])
        else:
            self.target_frame = frame[self.target_columns]
            self.classes = None
        self.feature_frame = frame[self.feature_columns]

    def _build_default_transform(self):
        normalize = tab_transforms.Normalize(self.feature_columns, self.feature_stats, self.normalization_mode)
        if not self.transform:
            if self.use_normalization:
                self.transform = transforms.Compose([
                    tab_transforms.NumpyToTensor(),
                    tab_transforms.ToFloatTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    tab_transforms.NumpyToTensor(),
                    tab_transforms.ToFloatTensor(),
                ])

        if not self.target_transform:
            if self.target_dtype == 'categorical':
                self.target_transform = transforms.Compose([
                    tab_transforms.NumpyToTensor(),
                    tab_transforms.ToLongTensor()
                ])
            else:
                self.target_transform = transforms.Compose([
                    tab_transforms.NumpyToTensor(),
                    tab_transforms.ToFloatTensor()
                ])

    def __getitem__(self, idx):
        feature = self.feature_frame.iloc[idx]
        feature = feature.values

        target = self.target_frame.iloc[idx]
        target = target.values

        if self.transform:
            feature = self.transform(feature)

        if self.target_transform:
            target = self.target_transform(target)

        if self.target_dtype == 'categorical':
            target = target.argmax(dim=0)

        return feature, target

    def sample(self, num=10, use_transform=False, to_tensor=True, shuffle=False):
        if not shuffle: random.seed(1261)

        len_list = list(range(len(self.feature_frame.values)))
        sample_idx = random.sample(len_list, num)

        feature = self.feature_frame.iloc[sample_idx]
        feature.index = pd.RangeIndex(len(feature.index))
        feature = feature.values

        target = self.target_frame.iloc[sample_idx]
        target.index = pd.RangeIndex(len(target.index))
        target = target.values

        if use_transform:
            feature = self.transform(feature)
            target = self.target_transform(target)
            if self.target_dtype == 'categorical':
                target = target.argmax(dim=1)

        if not use_transform and to_tensor:
            feature = torch.FloatTensor(feature)
            target = torch.Tensor(target)
            if self.target_dtype == 'categorical':
                target = target.argmax(dim=1)

        return feature, target

    def __len__(self):
        return len(self._proper_frame())
