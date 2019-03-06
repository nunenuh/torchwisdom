import torch
import torch.nn as nn


__all__ = ['UNet']


def conv_bn_relu(in_ch, out_ch, ksize=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch, out_ch),
            conv_bn_relu(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

    def forward(self, x):

        return x