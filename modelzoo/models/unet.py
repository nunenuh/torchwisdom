import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['UNet']


def conv_bn_relu(in_ch, out_ch, ksize=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def conv_bn_lrelu(in_ch, out_ch, ksize=3, padding=1, stride=1, neg_slope=0.2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(inplace=True, negative_slope=neg_slope)
    )

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu'):
        super(DoubleConv, self).__init__()
        if activation=='relu':
            self.conv = nn.Sequential(
                conv_bn_relu(in_ch, out_ch),
                conv_bn_relu(out_ch, out_ch)
            )
        elif activation=='lrelu':
            self.conv = nn.Sequential(
                conv_bn_lrelu(in_ch, out_ch),
                conv_bn_lrelu(out_ch, out_ch)
            )

    def forward(self, x):
        return self.conv(x)

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, pool='maxpool'):
        super(DownConv, self).__init__()

        if pool=='maxpool':
            self.down = nn.MaxPool2d(2)
        elif pool=='stride':
            self.down = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, padding=1)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, mode='bilinear'):
        super(UpConv, self).__init__()
        if mode=='bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif mode=='nearest':
            self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
        elif mode=='transpose':
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def _padfix(self, x1, x2):
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x1, x2

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1, x2 = self._padfix(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_chan, n_classes):
        super(UNet, self).__init__()
        self.inconv = InConv(in_chan, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 512)
        self.up1 = UpConv(1024, 256)
        self.up2 = UpConv(512, 128)
        self.up3 = UpConv(256, 64)
        self.up4 = UpConv(128, 64)
        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        inc = self.inconv(x)
        dc1 = self.down1(inc)
        dc2 = self.down2(dc1)
        dc3 = self.down3(dc2)
        dc4 = self.down4(dc3)
        up1 = self.up1(dc4, dc3)
        up2 = self.up2(up1, dc2)
        up3 = self.up3(up2, dc1)
        up4 = self.up4(up3, inc)
        out = self.outconv(up4)
        return out


if __name__ == '__main__':
    model = UNet(in_chan=1, n_classes=1)
    print(model)

    input = torch.rand(1,1,224,224)
    output = model(input)
    print(output)