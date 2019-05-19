from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import *


__all__ = ['DataCapsule']

class DataCapsule(object):
    r''''DataCapsule is an encapsulation of dataset and dataloader for Trainer Class'''
    def __init__(self, trainset: Dataset, validset: Dataset, testset: Dataset = None,
                 batch_size: int = 64, shuffle: bool = True, num_workers: int = 2):

        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.num_workers: int = num_workers

        self.trainset: Dataset = trainset
        self.validset: Dataset = validset
        self.testset: Dataset = testset

        self.train_loader: DataLoader = self._build_train_loader()
        self.valid_loader: DataLoader = self._build_valid_loader()
        self.test_loader: DataLoader = self._build_test_loader()

    def _create_loader(self, dataset):
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers)

    def _build_train_loader(self):
        return self._create_loader(self.trainset)

    def _build_valid_loader(self):
        return self._create_loader(self.validset)

    def _build_test_loader(self):
        if self.testset:
            return self._create_loader(self.testset)
        return None



if __name__ == "__main__":
    # dc = DatasetCollector()
    ...