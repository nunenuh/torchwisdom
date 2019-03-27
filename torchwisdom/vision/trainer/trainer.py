import torch
from torch.nn.modules.module import Module
from torch.optim import Optimizer
from torchwisdom.utils.data.collector import DatasetCollector
from torchwisdom.trainer import  Trainer


class ConvTrainer(Trainer):
    def __init__(self, data: DatasetCollector, model: Module,
                 optimizer: Optimizer, criterion: Module, metrics:dict,
                 callbacks=None, device='cpu'):
        '''

        :param data:
        :param model:
        :param optimizer:
        :param criterion:
        :param metrics:
        :param callbacks:
        :param device:

        '''
        super(ConvTrainer, self).__init__(data, model, optimizer, criterion, metrics, callbacks, device)
        self.bunch = self.data.bunch()
        self._set_device()

    def _set_device(self):
        self.model = self.model.to(device=self.device)
        # self.optimizer = self.optimizer.to(device=self.device)

    def build_optimizer(self, lr=0.001, mmt=0.9, wd=0.1):
        if self.optimizer is 'sgd':
            self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=mmt, weight_decay=wd)

    def train(self):
        train_loader = self.bunch['train']
        for idx, (img, label) in enumerate(train_loader):
            img = img.to(device=self.device)
            label = label.to(device=self.device)

            self.optimizer.zero_grad()
            out = self.model(img)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()

    def valid(self):
        valid_loader = self.bunch['valid']
        with torch.no_grad():
            for idx, (img, label) in enumerate(valid_loader):
                img = img.to(device=self.device)
                label = label.to(device=self.device)

                out = self.model(img)
                loss = self.criterion(out, label)

    def fit(self, num_epoch, lr=0.001, verbose=False):

        for epoch in range(num_epoch):
            self.train()
            self.valid()



if __name__ == '__main__':
    from torchwisdom.vision.models import  mobilenetv2
    from torchvision.datasets.mnist import MNIST
    import torchvision.transforms as transforms
    from torchwisdom.utils.data.collector import DatasetCollector
    import torch.optim as optim
    import torch.nn as nn

    train_path = '/data/MNIST/train'
    valid_path = '/data/MNIST/valid'

    tmft = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    trainset = MNIST(train_path, train=True, transform=tmft, download=True)
    validset = MNIST(valid_path, train=False, transform=tmft, download=True)

    data = DatasetCollector(trainset, validset)
    model = mobilenetv2(pretrained=False, in_chan=1)
    criterion = nn.CrossEntropyLoss()

    trainer = ConvTrainer(data=data, model=model, optimizer="sgd", criterion=criterion,
                          metrics=['accuracy_topk'])
    trainer.fit(1)

