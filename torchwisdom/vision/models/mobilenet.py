import torch
import torch.nn as nn
import math

__all__ = ['MobileNetV1','mobilenetv1','MobileNetV2','mobilenetv2']

model_urls = {
    'mobilenetv1':'',
    'mobilenetv2':'',
}

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=in_chan, bias=bias)

    def forward(self, input):
        return self.conv(input)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv3x3BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, strd=1, pad=1, bias=False):
        super(Conv3x3BnRelu, self).__init__()
        self.conv3x3 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=strd, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv3x3(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DWConv3x3BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, strd=1, pad=1, bias=False):
        super(DWConv3x3BnRelu, self).__init__()
        self.dwconv3x3 = DepthwiseConv2d(in_chan, out_chan, kernel_size=ksize, stride=strd, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.dwconv3x3(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1BnRelu(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=1, strd=1, pad=0, bias=False):
        super(Conv1x1BnRelu, self).__init__()
        self.conv1x1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=strd, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1x1(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, in_chan=3, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.cfg_pt1 = [(32,64,1), (64,128,2), (128,128,1), (128,256,2), (256,256,1), (256,512,2)]
        self.cfg_pt2 = [(512,512,1) for i in range(5)]
        self.cfg_pt3 = [(512,1024,2), (1024,1024,1)]
        self.config = self.cfg_pt1 + self.cfg_pt2 + self.cfg_pt3

        self.conv3x3bnrl = Conv3x3BnRelu(in_chan, 32, strd=2, pad=1)

        self.conv_layer = nn.ModuleList()
        for cfg in self.config:
            self.conv_layer.append(DWConv3x3BnRelu(cfg[0], cfg[0], strd=cfg[2]))
            self.conv_layer.append(Conv1x1BnRelu(cfg[0], cfg[1]))
        self.conv_layer = nn.Sequential(*self.conv_layer)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = Flatten()
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input):
        x = self.conv3x3bnrl(input)
        x = self.conv_layer(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



# this code below is taken from
# https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py
# at Thu Mar 7 2019
# ===================================================== start =========================================================#
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_chan=3, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(in_chan, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, self.output_channel)
        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.classifier = nn.Linear(self.output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ================================================ end ===============================================================#

def mobilenetv1(pretrained=True, in_chan=3, num_classes=1000):
    model = MobileNetV1(in_chan=in_chan, num_classes=num_classes)

    return model

def mobilenetv2(pretrained=True, in_chan=3, num_classes=1000):
    model = MobileNetV2(in_chan=in_chan, num_classes=num_classes)
    return model



if __name__ == '__main__':
    model = mobilenetv2()
    print(model)

    input = torch.rand(1,3,224,224)
    output = model(input)
    print(output)