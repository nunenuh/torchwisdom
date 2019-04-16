import torch
from torch.utils import data
import numpy as np
import pandas as pd
from torchwisdom.tabular import transforms

from pathlib import Path
import os

__all__ = ['CSVDataset']


class CSVDataset(data.dataset.Dataset):
    def __init__(self, file, target_column: list = None, feature_columns: list = None,
                 transform: object = None, target_transform: object = None):
        super(CSVDataset, self).__init__()
        self.file = Path(file)
        self.data_frame = self._file_check()
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.transform = transform
        self.target_transform = target_transform

    def _file_check(self):
        if not self.file.exists():
            raise ValueError('File not found!, please check your filepath')
        elif not self.file.is_file():
            raise ValueError('Expected file as input in parameter file but, got other type')

        return pd.read_csv(self.file)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

