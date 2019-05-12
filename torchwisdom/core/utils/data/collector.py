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
    def __init__(self, trainset: Dataset, validset: Dataset, testset: Dataset=None,
                 batch_size=64, shuffle=True, num_workers=2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.trainset: Dataset = trainset
        self.validset: Dataset = validset
        self.testset: Dataset = testset

        self._build_loader()



    def _build_loader(self):
        self.trainloader: DataLoader = DataLoader(self.trainset, batch_size=self.batch_size,
                                                  shuffle=self.shuffle, num_workers=self.num_workers)
        self.validloader: DataLoader = DataLoader(self.validset, batch_size=self.batch_size,
                                                  shuffle=self.shuffle, num_workers=self.num_workers)
        self.testloader: DataLoader = None
        if self.testset:
            self.testloader: DataLoader = DataLoader(self.testset, batch_size=self.batch_size,
                                                     shuffle=self.shuffle, num_workers=self.num_workers)

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
