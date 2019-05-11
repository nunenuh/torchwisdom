import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import squeezenet
import torch.utils.model_zoo as model_zoo
from torchwisdom.vision.models import mobilenet
from torchwisdom.core import nn as layers


__all__ = ['SiameseResNet','SiameseModelTrainer','siamese_resnet', 'siamese_mobilenet']


class SiameseResNet(resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(SiameseResNet, self).__init__(block, layers, num_classes)
        self.block_expansion = block.expansion

class SiameseMobileNetV2(mobilenet.MobileNetV2):
    def __init__(self, in_chan=3, num_classes=1000, input_size=224):
        super(SiameseMobileNetV2, self).__init__(in_chan, num_classes, input_size)

class SiameseSqueezeNet(squeezenet.SqueezeNet):
    def __init__(self, in_chan=3, version=1, num_classes=1000):
        super(SiameseSqueezeNet, self).__init__(version=version, num_classes=num_classes)
        self.num_classes = num_classes
        if version == 1.0:
            self.features[0] = nn.Conv2d(in_chan, 96, kernel_size=7, stride=2)
        else:
            self.features[0] = nn.Conv2d(in_chan, 64, kernel_size=3, stride=2)



class SiameseModelTrainer(nn.Module):
    def __init__(self, siamese_base):
        super(SiameseModelTrainer, self).__init__()
        self.siamese_base = siamese_base

    def forward_once(self, x):
        return self.siamese_base(x)

    def forward(self, x, y):
        output1 = self.forward_once(x)
        output2 = self.forward_once(y)
        return output1, output2


def siamese_resnet(pretrained_backbone=True, encoder_digit=64, version=18, in_chan=3, **kwargs):
    if in_chan != 3 and pretrained_backbone:
        raise ValueError("in_chan has to be 3 when you set pretrained=True")

    block = {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3], '50': [3, 4, 6, 3],
             '101': [3, 4, 23, 3], '152': [3, 8, 36, 3]}
    name_ver = 'resnet'+str(version)

    backbone = SiameseResNet(resnet.BasicBlock, block[str(version)], **kwargs)
    if pretrained_backbone:
        backbone.load_state_dict(model_zoo.load_url(resnet.model_urls[name_ver]))
    expansion = 512 * backbone.block_expansion
    backbone.fc = layers.Classfiers(in_features=expansion, n_classes=encoder_digit)
    model_trainer = SiameseModelTrainer(backbone)

    return model_trainer, backbone


def siamese_mobilenet(pretrained_backbone=True, encoder_digit=64, version=2, in_chan=3, input_size=224, **kwargs):
    if in_chan != 3 and input_size!=224 and pretrained_backbone:
        raise ValueError("in_chan has to be 3 and input_size has to be 224 when you set pretrained=True")

    backbone = SiameseMobileNetV2(in_chan=in_chan, input_size=input_size)
    if pretrained_backbone:
        mobilenetv2_url = 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-0c6065bc.pth'
        backbone.load_state_dict(model_zoo.load_url(mobilenetv2_url))
    backbone.classifier= layers.SimpleClassifiers(in_features=backbone.output_channel, n_classes=encoder_digit)
    model_trainer = SiameseModelTrainer(backbone)

    return model_trainer, backbone


def siamese_squeezenet(pretrained_backbone=True, encoder_digit=64, version=1.1, in_chan=3, **kwargs):
    if in_chan != 3 and pretrained_backbone:
        raise ValueError("in_chan has to be 3 when you set pretrained=True")

    name_ver = "squeezenet" + "_".join(str(version).split('.'))
    backbone = SiameseSqueezeNet(in_chan=in_chan, version=version, **kwargs)
    if pretrained_backbone:
        backbone.load_state_dict(model_zoo.load_url(squeezenet.model_urls[name_ver]))
    backbone.num_classes = encoder_digit
    backbone.classifier = layers.SqueezeNetCustomClassifers(num_classes=encoder_digit)
    model_trainer = SiameseModelTrainer(backbone)
    return model_trainer, backbone




if __name__ == '__main__':
    # resnet = torchvision.models.resnet18()
    trainer, backbone = siamese_squeezenet(pretrained_backbone=True, encoder_digit=32, version=1.0)
    backbone.eval()
    # print(backbone)
    x = torch.randn(1,3,224,224)
    pic1  = backbone(x)

    y = torch.randn(1, 3, 224, 224)
    pic2 = backbone(y)

    euc = F.pairwise_distance(pic1, pic2)
    print(euc)




