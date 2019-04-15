import torch
import torch.nn as nn
from torchvision.models.alexnet import AlexNet as VisionAlexNet
from torchvision.models.alexnet import model_urls
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


class AlexNet(VisionAlexNet):
    def __init__(self, in_feat=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features[0] = nn.Conv2d(in_feat, 64, kernel_size=11, stride=4, padding=2)


def alexnet(pretrained=True, in_feat=3, num_classes=1000, **kwargs):
    if pretrained and in_feat != 3:
        raise ValueError(f'when using pretrained=True, in_feat value is expected to be 3 but got {str(in_feat)}')

    net = AlexNet(in_feat=in_feat, **kwargs)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    if num_classes != 1000:
        net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )
    return net


if __name__ == '__main__':
    x = torch.rand(1,1,64,64)
    al = alexnet(pretrained=False, in_feat=1, num_classes=10)
    print(al(x))

