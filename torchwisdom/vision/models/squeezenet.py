import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models.squeezenet import SqueezeNet as VisionSqueezeNet
from torchvision.models.squeezenet import model_urls
import torch.utils.model_zoo as model_zoo

__all__ = []


class SqueezeNet(VisionSqueezeNet):
    def __init__(self, in_feat=3, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__(version, num_classes)
        if version == 1.0:
            self.features[0] = nn.Conv2d(in_feat, 96, kernel_size=7, stride=2)
        else:
            self.features[0] = nn.Conv2d(in_feat, 64, kernel_size=3, stride=2)

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def init_final_weight(self, final_conv):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is final_conv:
                init.normal_(m.weight, mean=0.0, std=0.01)


def squeezenet(pretrained=True, in_feat=3, version=1.1, num_classes=1000, **kwargs):
    if pretrained and in_feat != 3:
        raise ValueError(f'When using pretrained=True, in_feat value is expected to be 3 but got {str(in_feat)}')

    seqver = 'squeezenet'+'_'.join(str(version).split("."))
    net = SqueezeNet(in_feat=in_feat, version=version, **kwargs)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(model_urls[seqver]))

    if num_classes != 1000:
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        net.num_classes = num_classes
        net.init_final_weight(final_conv)

    return net


if __name__ == "__main__":
    x = torch.rand(1, 3, 64, 64)
    net = squeezenet(pretrained=True, in_feat=3, version=1.0, num_classes=10)
    out = net.forward(x)
    print(out)