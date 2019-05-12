import torch
from torch.utils import data
import numpy as np
import pandas as pd
from torchwisdom.tabular import transforms

from pathlib import Path
import os
from typing import *
from ..tabular import transforms as tab_transforms
import torchvision.transforms as transforms
import random
from ..core.nn import functional as N


__all__ = ['CSVDataset']


class CSVDataset(data.dataset.Dataset):
    def __init__(self, file, target_columns: Union[str, List], feature_columns: list = None,
                 transform: Any = None, target_transform: Any = None,
                 **kwargs):
        super(CSVDataset, self).__init__()
        self.file = Path(file)
        self.target_dtype = kwargs.get("target_dtype", "categorical")
        self.drop_columns = kwargs.get("drop_columns", [])
        self.data_frame = self._file_check()
        self.train_frame = None
        self.valid_frame = None
        self.target_columns = target_columns
        self.feature_columns = feature_columns
        self.transform: Callable = transform
        self.target_transform: Callable = target_transform
        self.valid_size = kwargs.get("valid_size", 0)
        self.mode = kwargs.get("mode", "train")
        self.use_normalization = kwargs.get("use_normalization", False)
        self.normalization_mode = kwargs.get('normalization_mode', 'minmax')
        self._build_classes()
        self._dataset_split()
        self._build_column()
        self._build_feature_stats()
        self._normalize_feature()
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
            xmin = frame[col].min()
            xmax = frame[col].max()
            xmean = frame[col].mean()
            xstd = frame[col].std()
            self.feature_stats.update({col: {'min': xmin, 'max': xmax, 'mean': xmean, 'std': xstd}})

    def _normalize_feature(self):
        if self.use_normalization:
            frame = self._proper_frame()
            for col in self.feature_columns:
                stats = self.feature_stats[col]
                if self.normalization_mode == 'minmax':
                    frame_result = N.normalization(self.data_frame[col].values, stats['min'], stats['max'])
                    train_result = N.normalization(self.train_frame[col].values, stats['min'], stats['max'])
                    valid_result = N.normalization(self.valid_frame[col].values, stats['min'], stats['max'])
                else:
                    frame_result = N.standardization(self.data_frame[col].values, stats['mean'], stats['std'])
                    train_result = N.standardization(self.train_frame[col].values, stats['mean'], stats['std'])
                    valid_result = N.standardization(self.valid_frame[col].values, stats['mean'], stats['std'])

                self.data_frame[col] = frame_result
                self.train_frame[col] = train_result
                self.valid_frame[col] = valid_result




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

            self.valid_frame = self.data_frame.iloc[valid_index]
            self.valid_frame.index = pd.RangeIndex(len(self.valid_frame.index))

            self.train_frame = self.data_frame.iloc[train_index]
            self.train_frame.index = pd.RangeIndex(len(self.train_frame.index))
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
        if not self.transform:
            self.transform = transforms.Compose([
                tab_transforms.NumpyToTensor(),
                tab_transforms.ToFloatTensor()
            ])

        if not self.target_transform:
            if self.target_dtype == 'categorical':
                self.target_transform =  transforms.Compose([
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
        feature = np.array(list(feature))
        feature = self.transform(feature)

        target = self.target_frame.iloc[idx]
        target = np.array(list(target))
        target = self.target_transform(target)
        if self.target_dtype == 'categorical':
            target = target.argmax(dim=0)
        return feature, target

    def __len__(self):
        return len(self._proper_frame())


