from torchwisdom.core.utils import DatasetCollector
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from torchwisdom.vision import transforms as ptransforms
from .datasets import CSVDataset


class CSVDataCollector(object):
    def __init__(self, path, batch_size=32, shuffle=True, num_workers=1, valid_size=0.2, **kwargs):
        self.path = Path(path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.target_column = kwargs.get("target_columns", None)
        self.feature_column = kwargs.get("feature_columns", None)
        self.target_dtype = kwargs.get("target_dtype", "categorical")
        self.drop_columns = kwargs.get("drop_columns", [])
        self.use_normalization = kwargs.get("use_normalization", False)
        self.normalization_mode = kwargs.get("normalization_mode", 'minmax')
        self._build_transform(**kwargs)
        self._build_dataset()
        self._build_collector()

    def _build_transform(self, **kwargs):
        self.trainset_transform = kwargs.get("trainset_transform", None)
        self.trainset_target_transform = kwargs.get("trainset_target_transform", None)
        self.validset_transform = kwargs.get("validset_transform", None)
        self.validset_target_transform = kwargs.get("validset_target_transform", None)

        if not self.trainset_transform:
            pass

    def _build_dataset(self):
        self.trainset = CSVDataset(str(self.path),
                                   target_columns=self.target_column,
                                   feature_columns=self.feature_column,
                                   mode='train',
                                   target_dtype=self.target_dtype,
                                   valid_size=self.valid_size,
                                   drop_columns=self.drop_columns,
                                   transform=self.trainset_transform,
                                   target_transform=self.trainset_target_transform,
                                   use_normalization=self.use_normalization,
                                   normalization_mode=self.normalization_mode,
                                   )
        self.validset = CSVDataset(str(self.path),
                                   target_columns=self.target_column,
                                   feature_columns=self.feature_column,
                                   mode='valid',
                                   target_dtype=self.target_dtype,
                                   valid_size=self.valid_size,
                                   drop_columns=self.drop_columns,
                                   transform=self.validset_transform,
                                   target_transform=self.validset_target_transform,
                                   use_normalization = self.use_normalization,
                                   normalization_mode = self.normalization_mode,
                                   )

    def _build_collector(self):
        self.data_collector = DatasetCollector(self.trainset, self.validset,
                                               batch_size=self.batch_size, shuffle=self.shuffle,
                                               num_workers=self.num_workers)

    def collector(self):
        return self.data_collector


