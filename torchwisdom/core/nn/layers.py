import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Flatten', 'AdaptiveConcatPool2d', 'Lambda', 'SimpleClassifiers', 'SqueezeNetCustomClassifers']


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


# got from fastai
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz:int=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


#got from fastai
class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)


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


class SimpleClassifiers(nn.Module):
    def __init__(self, in_features, n_classes, use_batchnorm=True, use_dropout=True, dprob=0.3):
        super(SimpleClassifiers, self).__init__()
        if use_batchnorm : self.bn = nn.BatchNorm1d(in_features)
        if use_dropout: self.dropout = nn.Dropout(p=dprob)
        self.fc = nn.Linear(in_features, n_classes)

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

    def forward(self, x):
        if self.use_batchnorm: x = self.bn(x)
        if self.use_dropout: x = self.dropout(x)
        x = self.fc(x)
        return x


class SqueezeNetCustomClassifiers(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNetCustomClassifiers, self).__init__()
        self.num_classes = num_classes
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifiers = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1),
        )

    def forward(self, x):
        x = self.classifiers(x)
        return x
