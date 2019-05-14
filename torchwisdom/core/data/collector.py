from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import *


__all__ = ['DatasetAttribute', 'DatasetCollector']


class DatasetAttribute(object):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @property
    def classes(self) -> Union[bool, List]:
        if hasattr(self._dataset, 'classes'):
            classes: List = self._dataset.classes
            return classes
        return False

    @property
    def class_to_idx(self) -> Union[bool, Dict]:
        if hasattr(self._dataset, 'class_to_idx'):
            class_to_idx: Dict = self._dataset.class_to_idx
            return class_to_idx
        return False

    @property
    def transform(self) -> object:
        if hasattr(self._dataset, 'transform'):
            transform = self._dataset.transform
            return transform
        return False


class DatasetCollector(object):
    def __init__(self, trainset: Dataset, validset: Dataset, testset: Dataset=None, **kwargs: Any):
        self.kwargs = kwargs
        self.batch_size = 64
        self.shuffle = True
        self.num_workers = 2
        self._build_base()

        self.trainset: Dataset = trainset
        self.validset: Dataset = validset
        self.testset: Dataset = testset

        self.trainloader: DataLoader = None
        self.validloader: DataLoader = None
        self.testloader: DataLoader = None
        self._build_loader()

        self.transform = None
        self._build_transform()

        self.input_size = ()
        self.data_type = 'classification'
        self.case_type = 'vision'
        self._build_metadata()

    def _build_base(self):
        if 'shuffle' in self.kwargs:
            self.shuffle = self.kwargs.get("shuffle")
        if 'num_workers' in self.kwargs:
            self.num_workers = self.kwargs.get("num_workers")
        if 'batch_size' in self.kwargs:
            self.batch_size = self.kwargs.get("batch_size")

    def _build_metadata(self):
        feat, targ = self.validset.__getitem__(0)
        self.input_size = feat.size()
        if 'data_type' in self.kwargs:
            self.data_type = self.kwargs['data_type']
        if 'case_type' in self.kwargs:
            self.case_type = self.kwargs['case_type']

    def _build_train_loader(self):
        self.trainloader = DataLoader(dataset=self.trainset,
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=self.num_workers)

    def _build_valid_loader(self):
        self.validloader = DataLoader(dataset=self.validset,
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=self.num_workers)

    def _build_test_loader(self):
        if self.testset:
            self.testloader = DataLoader(dataset=self.testset,
                                         batch_size=self.batch_size,
                                         shuffle=self.shuffle,
                                         num_workers=self.num_workers)

    def _build_loader(self):
        self._build_train_loader()
        self._build_valid_loader()
        self._build_test_loader()

    def _build_transform(self, validset_transform=True):
        if validset_transform:
            self.transform = self.validset_attr.transform
        else:
            self.transform = self.trainset_attr.transform

    @property
    def trainset_attr(self) -> DatasetAttribute:
        return DatasetAttribute(self.trainset)

    @property
    def validset_attr(self) -> DatasetAttribute:
        return DatasetAttribute(self.validset)

    def bunch(self):
        return {'train': self.trainloader,
                'valid': self.validloader,
                'test': self.testloader}


if __name__ == "__main__":
    dc = DatasetCollector()
