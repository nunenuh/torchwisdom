import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import  BasicBlock, Bottleneck, model_urls, ResNet
import torch.utils.model_zoo as model_zoo

__all__ = ['SiameseResNet','SiameseTrainer','siamese_resnet18','siamese_resnet34','siamese_resnet50','siamese_resnet101','siamese_resnet152']

class Classfiers(nn.Module):
    def __init__(self, in_features, n_classes, use_batchnorm=True, use_dropout=True, dprob=[0.5,0.3,0.2]):
        super(Classfiers, self).__init__()
        modules = []
        if use_batchnorm: modules.append(nn.BatchNorm1d(in_features))
        if use_dropout: modules.append(nn.Dropout(dprob[0]))
        modules.append(nn.Linear(in_features, in_features // 2))
        modules.append(nn.ReLU(inplace=True))

        if use_batchnorm: modules.append(nn.BatchNorm1d(in_features//2))
        if use_dropout: modules.append(nn.Dropout(dprob[1]))
        modules.append(nn.Linear(in_features //2, in_features // 4))
        modules.append(nn.ReLU(inplace=True))

        if use_batchnorm: modules.append(nn.BatchNorm1d(in_features//4))
        if use_dropout: modules.append(nn.Dropout(dprob[2]))
        modules.append(nn.Linear(in_features //4, n_classes))

        self.classfiers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.classfiers(x)
        return x


class SiameseResNet(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(SiameseResNet, self).__init__(block, layers, num_classes)
        self.block_expansion = block.expansion


class SiameseTrainer(nn.Module):
    def __init__(self, siamese_base):
        super(SiameseTrainer, self).__init__()
        self.siamese_base = siamese_base

    def forward_once(self, x):
        return self.siamese_base(x)

    def forward(self, x, y):
        output1 = self.forward_once(x)
        output2 = self.forward_once(y)
        return output1, output2


def siamese_resnet18(pretraned_backbone=True, encoder_digit=64, **kwargs):
    backbone = SiameseResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretraned_backbone:
        backbone.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    expansion = 512 * backbone.block_expansion
    backbone.fc = Classfiers(in_features=expansion, n_classes=encoder_digit)
    model_trainer = SiameseTrainer(backbone)

    return model_trainer, backbone


def siamese_resnet34(pretraned_backbone=True, encoder_digit=64, **kwargs):
    backbone = SiameseResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretraned_backbone:
        backbone.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    expansion = 512 * backbone.block_expansion
    backbone.fc = Classfiers(in_features=expansion, n_classes=encoder_digit)
    model_trainer = SiameseTrainer(backbone)

    return model_trainer, backbone


def siamese_resnet50(pretraned_backbone=True, encoder_digit=64, **kwargs):
    backbone = SiameseResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretraned_backbone:
        backbone.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    expansion = 512 * backbone.block_expansion
    backbone.fc = Classfiers(in_features=expansion, n_classes=encoder_digit)
    model_trainer = SiameseTrainer(backbone)

    return model_trainer, backbone


def siamese_resnet101(pretraned_backbone=True, encoder_digit=64, **kwargs):
    backbone = SiameseResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretraned_backbone:
        backbone.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    expansion = 512 * backbone.block_expansion
    backbone.fc = Classfiers(in_features=expansion, n_classes=encoder_digit)
    model_trainer = SiameseTrainer(backbone)

    return model_trainer, backbone


def siamese_resnet152(pretraned_backbone=True, encoder_digit=64, **kwargs):
    backbone = SiameseResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretraned_backbone:
        backbone.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    expansion = 512 * backbone.block_expansion
    backbone.fc = Classfiers(in_features=expansion, n_classes=encoder_digit)
    model_trainer = SiameseTrainer(backbone)

    return model_trainer, backbone


if __name__ == '__main__':
    # resnet = torchvision.models.resnet18()
    trainer, backbone = siamese_resnet18()
    print(backbone)
    x = torch.randn(2,3,224,224)
    print(backbone(x).shape)

