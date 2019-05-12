from torchwisdom.core.utils import DatasetCollector
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from torchwisdom.vision import transforms as ptransforms



class ImageClassifierData(object):
    def __init__(self, path, batch_size=32, shuffle=True, num_workers=2,
                 image_size=(224,224), device='cpu'):
        self.path = Path(path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.image_size = image_size
        self.device = device
        self._build_transform()
        self._build_dataset()
        self._build_data_collector()

    def _build_transform(self):
        imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.trainset_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalize
        ])

        self.validset_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            imagenet_normalize
        ])

    def _build_dataset(self):
        self.trainpath = self.path.joinpath('train')
        self.validpath = self.path.joinpath('valid')
        self.trainset = datasets.ImageFolder(str(self.trainpath), transform=self.trainset_transform)
        self.validset = datasets.ImageFolder(str(self.validpath), transform=self.validset_transform)

    def _build_data_collector(self):
        self.data_collector = DatasetCollector(self.trainset, self.validset,
                                               batch_size=self.batch_size, shuffle=self.shuffle,
                                               num_workers=self.num_workers)

    def collector(self):
        return self.data_collector


class ImageAutoEncoderData(object):
    def __init__(self, path, batch_size=32, shuffle=True, num_workers=2,
                 image_size=(224, 224), device='cpu'):
        self.path = Path(path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.image_size = image_size
        self.device = device
        self._build_transform()
        self._build_dataset()
        self._build_data_collector()

    def _build_transform(self):
        imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trainset_pair_transform = ptransforms.PairCompose([
            ptransforms.PairResize((64)),
            ptransforms.PairCenterCrop((64)),
            ptransforms.PairRandomHorizontalFlip(),
            # ptransforms.PairRandomVerticalFlip(),
            ptransforms.PairRandomRotation(10),
            ptransforms.PairToTensor(),
        ])

        self.trainset_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalize
        ])


        self.validset_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            imagenet_normalize
        ])

    def _build_dataset(self):
        self.trainpath = self.path.joinpath('train')
        self.validpath = self.path.joinpath('valid')
        self.trainset = datasets.ImageFolder(str(self.trainpath), transform=self.trainset_transform)
        self.validset = datasets.ImageFolder(str(self.validpath), transform=self.validset_transform)

    def _build_data_collector(self):
        self.data_collector = DatasetCollector(self.trainset, self.validset,
                                               batch_size=self.batch_size, shuffle=self.shuffle,
                                               num_workers=self.num_workers)

    def collector(self):
        return self.data_collector