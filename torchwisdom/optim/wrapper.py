import torch.optim as optim
import torch.nn as nn
import collections
from typing import *

__all__ = []


class ModelParser:
    @staticmethod
    def get_name(name_split):
        if len(name_split)<=2:
            key = name_split[-2]
        else:
            key = '.'.join(name_split[:-1])
        return key

    @staticmethod
    def get_params_group(model):
        layer_group = collections.OrderedDict()
        param_group, last_name = [], ""
        for idx, (name, param) in enumerate(model.named_parameters()):
            name_split = name.split(".")
            layer_name, name_type = name_split[-2], name_split[-1]
            key = ModelParser.get_name(name_split)
            if idx == 0:
                param_group.append(param)
                layer_group[key] = param_group
                last_name = layer_name
            else:
                if last_name == layer_name:
                    param_group.append(param)
                    layer_group[key] = param_group
                else:
                    param_group = []
                    param_group.append(param)
                    layer_group[key] = param_group
            last_name = layer_name
        return layer_group


class OptimizerWrapper(object):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer = None):
        self.model = model
        self.optimizer: optim.Optimizer = optimizer

        self.lr_decay = 0
        self.momentum = 0
        self.beta1, self.beta2 = 0.9, 0.999
        self.betas = [self.beta1, self.beta2]
        self.weight_decay = 0
        self.eps = 1e-08
        self.alpha = 0.99
        self.dampening = 0
        self.nesterov = False

    def create(self, lr: Union[float, list], name='sgd', **kwargs):
        model_params = self._model_parameters(lr)
        if self.optimizer:
            return self.optimizer(model_params)
        else:
            self.optimizer = optim.Adam
            return self.optimizer(model_params)

    def _model_parameters(self, lr: Union[float, list], **kwargs):
        parameters = self._model_param_group()
        if type(lr) == list:
            build = []
            lr_len = len(lr)
            param_len = len(parameters)
            lr_every = None
            if param_len < lr_len:
                raise ValueError('lr list is more bigger than model parameter, try to remove item on list!')
            else:
                lr_every = param_len // len(lr)

            lr_idx = 0
            for idx, param in enumerate(parameters):
                if idx % lr_every == 0 and idx != 0:
                    lr_idx += 1
                if len(lr) <= lr_idx:
                    lr_idx -= 1
                #                 print(idx, lr[lr_idx])
                par = {'params': param, 'lr': lr[lr_idx]}
                build.append(par)
            return build
        else:
            return parameters

    def _lr_every(self, param, lr):
        param_len = len(param)
        lr_every = param_len // len(lr)
        lr_list = []
        for idx in range(param_len):
            if idx % lr_every == 0 and id != 0:
                lr_list.append(idx)
        return lr_list

    def _model_param_group_odict(self):
        return ModelParser.get_params_group(self.model)

    def _model_param_group(self):
        parameters = []
        for idx, (name, param) in enumerate(self._model_param_group_odict().items()):
            parameters.append(param)
        return parameters


def adam_builder(model: nn.Module, lr=0.001, **kwargs):
    optim.Adam(model.parameters(), lr=lr)




if __name__ == '__main__':
    from torchvision import models
    test_model = models.alexnet()
    opt = optim.Adam
    optwr = OptimizerWrapper(test_model)
    myoptim = optwr.create(lr=[0.1], betas=[0.5, 0.999])
    print(myoptim)



