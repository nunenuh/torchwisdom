import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TinyDarknet']

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class DarknetConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, padding=1, activation='lrelu', neg_slope=0.25):
        super(DarknetConvBlock, self).__init__()

        modules = []
        modules.append(nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=padding))
        modules.append(nn.BatchNorm2d(out_chan))

        if activation=='lrelu':
            modules.append(nn.LeakyReLU(inplace=True, negative_slope=neg_slope))
        elif activation=='relu':
            modules.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x)
        return x


class DarknetBottleNeck(nn.Module):
    def __init__(self, in_chan, hid_chan, out_chan, use_pool=True):
        super(DarknetBottleNeck, self).__init__()
        modules = []
        modules.append(DarknetConvBlock(in_chan, hid_chan, ksize=1, padding=0))
        modules.append(DarknetConvBlock(hid_chan, out_chan, ksize=3))
        modules.append(DarknetConvBlock(out_chan, hid_chan, ksize=1, padding=0))
        modules.append(DarknetConvBlock(hid_chan, out_chan, ksize=3))
        if use_pool:
            modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.bottleneck = nn.Sequential(*modules)

    def forward(self, x):
        x = self.bottleneck(x)
        return x

class TinyDarknet(nn.Module):
    def __init__(self, in_chan=3, n_classes=1000, input_size=224):
        super(TinyDarknet, self).__init__()

        self.feature = nn.Sequential(
            DarknetConvBlock(in_chan, 16, ksize=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DarknetConvBlock(16, 32, ksize=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.bottleneck1 = DarknetBottleNeck(32, 16, 128, use_pool=True)
        self.bottleneck2 = DarknetBottleNeck(128, 32, 256, use_pool=True)
        self.bottleneck3 = nn.Sequential(
            DarknetBottleNeck(256, 64, 512, use_pool=False),
            DarknetConvBlock(512, 128, ksize=1, stride=1, padding=0),
            DarknetConvBlock(128, n_classes, ksize=1, stride=1, padding=0,  activation='relu')
        )
        self.flatten = Flatten()
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(n_classes*(input_size//2**4)*(input_size//2**4), n_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        # x = self.fc(x)
        return x

def tinydarknet(pretrained=True, n_classes=1000, input_size=224):
    model = TinyDarknet(in_chan=3, n_classes=1000, input_size=224)

    return model


if __name__ == '__main__':
    model = tinydarknet()
    print(model)

    input = torch.rand(4,3,224,224)
    output = model(input)
    print(output.shape)











