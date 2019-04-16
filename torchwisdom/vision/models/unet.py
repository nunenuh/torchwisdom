import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo


__all__ = ['UNet', 'TuneableUNet']


# this code is inspired from  milesial repository
# https://github.com/milesial/Pytorch-UNet
# some of the code is taken from that repo and I make changes
# for the thing I need to change


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

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
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
    def __init__(self, in_ch, out_ch, use_upsample=False, mode='bilinear'):
        super(OutConv, self).__init__()
        self.use_upsample = use_upsample
        if use_upsample:
            if mode == 'bilinear':
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            elif mode == 'nearest':
                self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
            elif mode == 'transpose':
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        if self.use_upsample:
            x  = self.up(x)
        x = self.conv(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_chan, start_feat=64):
        super(UNetEncoder, self).__init__()
        self.out_chan = start_feat * 8
        self.inconv = InConv(in_chan, start_feat)
        self.down1 = DownConv(start_feat, start_feat*2)
        self.down2 = DownConv(start_feat*2, start_feat*4)
        self.down3 = DownConv(start_feat*4, start_feat*8)
        self.down4 = DownConv(start_feat*8, start_feat*8)

    def forward(self, x):
        inc = self.inconv(x)
        dc1 = self.down1(inc)
        dc2 = self.down2(dc1)
        dc3 = self.down3(dc2)
        dc4 = self.down4(dc3)
        return dc4, dc3, dc2, dc1, inc


class UNetDecoder(nn.Module):
    def __init__(self, in_chan, n_classes):
        super(UNetDecoder, self).__init__()
        self.up1 = UpConv(in_chan, in_chan//4)
        self.up2 = UpConv(in_chan//2, in_chan//8)
        self.up3 = UpConv(in_chan//4, in_chan//16)
        self.up4 = UpConv(in_chan//8, in_chan//16)
        self.outconv = OutConv(in_chan//16, n_classes)

    def forward(self, dc4, dc3, dc2, dc1, inc):
        up1 = self.up1(dc4, dc3)
        up2 = self.up2(up1, dc2)
        up3 = self.up3(up2, dc1)
        up4 = self.up4(up3, inc)
        out = self.outconv(up4)
        return out


class UNet(nn.Module):
    def __init__(self, in_chan, n_classes, start_feat=64):
        super(UNet, self).__init__()
        self.encoder_in_chan = in_chan
        self.decoder_in_chan = start_feat * 16
        self.start_feat = start_feat

        self.encoder = UNetEncoder(in_chan=self.encoder_in_chan, start_feat=start_feat)
        self.decoder = UNetDecoder(in_chan=self.decoder_in_chan, n_classes=n_classes)

    def forward(self, x):
        dc4, dc3, dc2, dc1, inc = self.encoder(x)
        out = self.decoder(dc4, dc3, dc2, dc1, inc)
        return out


class TuneableUNetEncoder(nn.Module):
    def __init__(self, in_chan, start_feat=64, deep=4):
        super(TuneableUNetEncoder, self).__init__()
        self.out_chan = start_feat * 8
        self.in_chan = in_chan
        self.start_feat = start_feat
        self.deep = deep
        self.chan = self._generate_chan()
        self.last_chan = self.chan[-1][-1]

        modules = self._make_layer(self.in_chan, self.start_feat)
        self.encoder = nn.Sequential(*modules)

    def _make_layer(self, in_chan, start_feat):
        modules = []
        modules.append(InConv(in_chan, start_feat))
        for d in range(self.deep):
           (in_chan, out_chan) = self.chan[d]
           modules.append(DownConv(in_chan, out_chan))
        return modules

    def _generate_chan(self):
        chan = []
        for d in range(self.deep):
            c1 = 2 ** d
            c2 = 2 ** (d + 1)
            if (d + 1) != self.deep:
                pair = (self.start_feat * c1, self.start_feat * c2)
            else:
                pair = (self.start_feat * c1, self.start_feat * c1)
            chan.append(pair)
        return chan

    def forward(self, x):
        output = []
        for d in range(self.deep+1):
            x = self.encoder[d](x)
            output.append(x)
        return output


class TuneableUNetDecoder(nn.Module):
    def __init__(self, in_chan, n_classes, deep=4):
        super(TuneableUNetDecoder, self).__init__()
        self.in_chan = in_chan
        self.n_classes = n_classes
        self.deep = deep
        self.chan = self._generate_chan()
        self.last_chan = self.chan[-1][-1]

        modules = self._make_layer()
        self.decoder = nn.Sequential(*modules)

    def _make_layer(self):
        modules = []
        for d in range(self.deep):
           (in_chan, out_chan) = self.chan[d]
           modules.append(UpConv(in_chan, out_chan))
        (in_chan, out_chan) = self.chan[self.deep]
        modules.append(OutConv(in_chan, self.n_classes))
        return modules

    def _generate_chan(self):
        chan = []
        self.in_chan = self.in_chan * 2
        for d in range(self.deep):
            c1 = self.in_chan // 2 ** (d)
            if (d + 1) != self.deep:
                c2 = self.in_chan // 2 ** (d + 2)
            else:
                c2 = self.in_chan // 2 ** (d + 1)
            pair = (c1, c2)
            chan.append(pair)

        output_pair = (c2, self.n_classes)
        chan.append(output_pair)
        return chan

    def forward(self, input):
        input.reverse()
        x = self.decoder[0](input[0], input[1])
        for i in range(1, self.deep+1):
            if i+1 != self.deep+1:
                x = self.decoder[i](x, input[i+1])
            else:
                x = self.decoder[self.deep](x)
        return x


class TuneableUNet(nn.Module):
    def __init__(self, in_chan, n_classes, config):
        super(TuneableUNet, self).__init__()
        self.config = config
        self.encoder = TuneableUNetEncoder(
            in_chan=in_chan,
            start_feat=self.config['start_feat'],
            deep=self.config['deep'],
        )

        self.decoder = TuneableUNetDecoder(
            in_chan=self.encoder.last_chan,
            n_classes=n_classes,
            deep=self.config['deep']
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResNetUNetEncoder(resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000, in_chan=3):
        super(ResNetUNetEncoder, self).__init__(block, layers, num_classes)
        self.expansion = block.expansion
        self.last_chan = block.expansion * 512
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        inc = x

        x = self.maxpool(x)
        x = self.layer1(x)
        dc1 = x

        x = self.layer2(x)
        dc2 = x

        x = self.layer3(x)
        dc3 = x

        x = self.layer4(x)
        dc4 =x

        return dc4, dc3, dc2, dc1, inc

class ResNetUNetDecoder(nn.Module):
    def __init__(self, in_chan, expansion, n_classes):
        super(ResNetUNetDecoder, self).__init__()
        self.in_chan = in_chan
        self.expansion = expansion
        self.n_classes = n_classes
        self.chan = self._generate_chan()

        self.up1 = UpConv(self.chan[0]+self.chan[1], 256)
        self.up2 = UpConv(self.chan[2]+256, 128)
        self.up3 = UpConv(self.chan[3]+128, 64)
        self.up4 = UpConv(self.chan[4]+64, 64)
        self.outconv = OutConv(64, n_classes, use_upsample=True)

    def _generate_chan(self):
        chan = []
        tmp_chan = self.in_chan
        for i in range(5):
            chan.append(tmp_chan)
            if self.expansion == 4:
                if i + 1 == 4:
                    tmp_chan = tmp_chan // 4
                else:
                    tmp_chan = tmp_chan // 2
            elif self.expansion == 1:
                if i + 1 == 4:
                    tmp_chan = tmp_chan
                else:
                    tmp_chan = tmp_chan // 2
        return chan

    def forward(self, dc4, dc3, dc2, dc1, inc):
        up1 = self.up1(dc4, dc3)
        up2 = self.up2(up1, dc2)
        up3 = self.up3(up2, dc1)
        up4 = self.up4(up3, inc)
        out = self.outconv(up4)
        return out

class ResNetUNet(nn.Module):
    def __init__(self, in_chan, n_classes, pretrained=True, version=18):
        super(ResNetUNet, self).__init__()
        self.pretrained = pretrained
        self.version = version
        self.in_chan = in_chan

        if in_chan!=3 and pretrained==True:
            raise ValueError("in_chan has to be 3 when you set pretrained=True")

        self.encoder = self._build_resnet()
        self.decoder = ResNetUNetDecoder(in_chan=self.encoder.last_chan, expansion=self.encoder.expansion, n_classes=n_classes)

    def _build_resnet(self):
        block = self._get_block()
        ver = self.version
        name_ver = 'resnet'+str(ver)
        if ver>=50:
            model = ResNetUNetEncoder(resnet.Bottleneck, block[str(ver)], in_chan=self.in_chan)
        else:
            model = ResNetUNetEncoder(resnet.BasicBlock, block[str(ver)], in_chan=self.in_chan)
        if self.pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls[name_ver]))
        return model

    def _get_block(self):
        return {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3],
                '50': [3, 4, 6, 3], '101': [3, 4, 23, 3],
                '152': [3, 8, 36, 3]}

    def forward(self, x):
        dc4, dc3, dc2, dc1, inc = self.encoder(x)
        out = self.decoder(dc4, dc3, dc2, dc1, inc)
        return out

if __name__ == '__main__':
    model = ResNetUNet(in_chan=3, n_classes=1, pretrained=True, version=18)
    input = torch.rand(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

    # expansion = 4
    # input = torch.rand(1, 3, 224, 224)
    # net = resnet.resnet50()
    # x = net.conv1(input)
    # x = net.bn1(x)
    # x = net.relu(x)
    # print('inconv',x.shape)
    # inconv = x
    #
    # x = net.maxpool(x)
    # x = net.layer1(x)
    # print('layer1',x.shape)
    # dc1 = x
    #
    # x  = net.layer2(x)
    # print('layer2',x.shape)
    # dc2 = x
    #
    # x = net.layer3(x)
    # print('layer3',x.shape)
    # dc3=x
    #
    # x = net.layer4(x)
    # print('layer4',x.shape)
    # dc4=x
    #
    # up1 = UpConv(dc4.size(1)+dc3.size(1), 512)(dc4, dc3)
    # print(up1.shape)
    #
    # up2 = UpConv(up1.size(1)+dc2.size(1), 256)(up1, dc2)
    # print(up2.shape)
    #
    # up3 = UpConv(up2.size(1)+dc1.size(1), 128)(up2, dc1)
    # print(up3.shape)
    #
    # up4 = UpConv(up3.size(1) + inconv.size(1), 64)(up3, inconv)
    # print(up4.shape)
    #
    # outconv = OutConv(64, 1, use_upsample=True)(up4)
    # print(outconv.shape)