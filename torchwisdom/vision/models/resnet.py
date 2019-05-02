import torch.nn as nn
from torchvision import models
import torchwisdom.core.nn.layers as layers
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet18','resnet34','resnet50','resnet101','resnet152']


class ResNet(models.resnet.ResNet):
    def __init__(self, block, layers, in_feat=3, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes=1000)
        self.block_expansion = block.expansion
        self.conv1 = nn.Conv2d(in_feat, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)


def resnet(pretrained=True, version=18, in_chan=3, num_classes=1000, **kwargs):
    if in_chan != 3 and pretrained:
        raise ValueError("in_chan has to be 3 when you set pretrained=True")

    block = {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3], '50': [3, 4, 6, 3],
             '101': [3, 4, 23, 3], '152': [3, 8, 36, 3]}
    name_ver = 'resnet' + str(version)
    net = ResNet(models.resnet.BasicBlock, block[str(version)], in_feat=in_chan, **kwargs)

    if pretrained and in_chan == 3:
        net.load_state_dict(model_zoo.load_url(models.resnet.model_urls[name_ver]))

    if num_classes != 1000:
        expansion = 512 * net.block_expansion
        net.fc = layers.Classfiers(in_features=expansion, n_classes=num_classes)
    return net


def resnet18(pretrained=True, in_chan=3, num_classes=1000, **kwargs):
    return resnet(pretrained=pretrained, version=18, in_chan=in_chan, num_classes=num_classes, **kwargs)


def resnet34(pretrained=True, in_chan=3, num_classes=1000, **kwargs):
    return resnet(pretrained=pretrained, version=34, in_chan=in_chan, num_classes=num_classes, **kwargs)


def resnet50(pretrained=True, in_chan=3, num_classes=1000, **kwargs):
    return resnet(pretrained=pretrained, version=50, in_chan=in_chan, num_classes=num_classes, **kwargs)


def resnet101(pretrained=True, in_chan=3, num_classes=1000, **kwargs):
    return resnet(pretrained=pretrained, version=101, in_chan=in_chan, num_classes=num_classes, **kwargs)


def resnet152(pretrained=True, in_chan=3, num_classes=1000, **kwargs):
    return resnet(pretrained=pretrained, version=152, in_chan=in_chan, num_classes=num_classes, **kwargs)
