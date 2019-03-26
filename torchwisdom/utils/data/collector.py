from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class DatasetCollector(object):
    def __init__(self, trainset:Dataset, validset:Dataset, testset:Dataset=None,
                 batch_size=64, shuffle=True, num_workers=2):
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def bunch(self):
        trainloader = DataLoader(self.trainset, batch_size=self.batch_size,
                                 shuffle=self.shuffle, num_workers=self.num_workers)
        validloader = DataLoader(self.validset, batch_size=self.batch_size,
                                 shuffle=self.shuffle, num_workers=self.num_workers)
        testloader = None
        if self.testset:
            testloader = DataLoader(self.testset, batch_size=self.batch_size,
                                    shuffle=self.shuffle, num_workers=self.num_workers)

        return {'train': trainloader,
                'valid': validloader,
                'test': testloader}



