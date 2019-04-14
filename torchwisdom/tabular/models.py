import torch
import torch.nn as nn


__all__ = ['SimpleNet', 'MultilayerNet']


class SimpleNet(nn.Module):
    def __init__(self, num_in_feat, num_hid_feat, num_out_feat):
        super(SimpleNet, self).__init__()
        self.input_layer = nn.Linear(num_in_feat, num_hid_feat)
        self.activation = nn.ReLU(inplace=True)
        self.output_layer = nn.Linear(num_hid_feat, num_out_feat)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


class MultilayerNet(nn.Module):
    def __init__(self, num_in_feat, num_hid_layer, num_hid_feat, num_out_feat):
        super(MultilayerNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_in_feat, num_hid_feat),
            nn.ReLU(inplace=True)
        )

        modules = []
        for idx in range(num_hid_layer):
            modules.append(nn.Linear(num_hid_feat, num_hid_feat))
            modules.append(nn.ReLU(inplace=True))
        self.hidden_layer = nn.Sequential(*modules)

        self.output_layer = nn.Linear(num_hid_feat, num_out_feat)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x





