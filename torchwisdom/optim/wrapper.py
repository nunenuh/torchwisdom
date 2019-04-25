import torch.optim as optim
import torch.nn as nn
import collections
from typing import *
from dataclasses import dataclass


__all__ = ['ModelParser', 'OptimizerWrapper', 'optim_builder', 'string_to_optim', 'optim_name']


class ModelParser:
    @staticmethod
    def get_name(name_split):
        if len(name_split)<=2:
            key = name_split[-2]
        else:
            key = '.'.join(name_split[:-1])
        return key

    @staticmethod
    def get_params_group_odict(model):
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

    @staticmethod
    def get_params_group(model):
        parameters = []
        for idx, (name, param) in enumerate(ModelParser.get_params_group_odict(model).items()):
            parameters.append(param)
        return parameters


class OptimizerWrapper(object):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer = None):
        self.model = model
        self.optimizer: optim.Optimizer = optimizer

    def create(self, lr: Union[float, list], opt: Union[str, optim.Optimizer] = None, **kwargs):
        model_params = self._model_parameters(lr)
        opt_build = None
        if type(opt) == str:
            # if opt is string than build optimizer from string
            opt_class = string_to_optim(opt)
            opt_build = optim_builder(opt_class, model_params, lr=lr, **kwargs)
        elif type(opt) == type:
            # if optimizer is class
            opt_build = optim_builder(opt, model_params, lr=lr, **kwargs)
        elif isinstance(opt, optim.Optimizer):
            # if optimizer is an instantiate object of Optimizer
            if hasattr(opt, 'defaults'):
                opt.defaults['lr'] = lr
            opt_build = opt
        else:
            #default opt
            opt_build = optim_builder(optim.Adam, model_params, lr=lr, **kwargs)

        self.optimizer = opt_build
        return self.optimizer

    def _model_parameters(self, lr: Union[float, list], **kwargs):
        parameters = ModelParser.get_params_group(self.model)
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
            return self.model.parameters()

    def _lr_every(self, param, lr):
        param_len = len(param)
        lr_every = param_len // len(lr)
        lr_list = []
        for idx in range(param_len):
            if idx % lr_every == 0 and id != 0:
                lr_list.append(idx)
        return lr_list


def optim_builder(opt:Any, parameters: List, **kwargs: Any):
    name = optim_name(opt)
    OptimClass = opt
    opt_build = None
    if name == "sgd":
        momentum = kwargs.get('momentum',0)
        dampening = kwargs.get('dampening', 0)
        weight_decay = kwargs.get('weight_decay', 0)
        nesterov = kwargs.get('nesterov', False)
        lr = kwargs.get('lr', 0.01)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.SGD = OptimClass(parameters, lr=lr, momentum=momentum,dampening=dampening,
                                         weight_decay=weight_decay, nesterov=nesterov)
    elif name == "asgd":
        lambd = kwargs.get('lambd', 1e-4)
        alpha = kwargs.get('alpha', 0.75)
        t0 = kwargs.get('t0', 1e6)
        weight_decay = kwargs.get('weight_decay', 0)
        lr = kwargs.get('lr', 1e-2)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.ASGD = OptimClass(parameters, lr=lr, lambd=lambd, alpha=alpha,
                                           t0=t0, weight_decay=weight_decay)
    elif name == "adam":
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        amsgrad= kwargs.get('amsgrad', 1e6)
        weight_decay = kwargs.get('weight_decay', 0)
        lr = kwargs.get('lr', 1e-3)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.Adam = OptimClass(parameters, lr=lr, betas=betas, eps=eps,
                                           amsgrad=amsgrad, weight_decay=weight_decay)
    elif name == "adamax":
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        weight_decay = kwargs.get('weight_decay', 0)
        lr = kwargs.get('lr', 2e-3)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.Adamax = OptimClass(parameters, lr=lr, betas=betas,
                                             eps=eps, weight_decay=weight_decay)
    elif name == "adadelta":
        rho = kwargs.get('rho', 0.9)
        eps = kwargs.get('eps', 1e-6)
        weight_decay = kwargs.get('weight_decay', 0)
        lr = kwargs.get('lr', 1.0)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.Adadelta = OptimClass(parameters, lr=lr, rho=rho,
                                               eps=eps, weight_decay=weight_decay)
    elif name == "adagrad":
        lr_decay = kwargs.get('lr_decay', 0.9)
        iav = kwargs.get('initial_accumulator_value', 0)
        weight_decay = kwargs.get('weight_decay', 0)
        lr = kwargs.get('lr', 1e-2)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.Adagrad = OptimClass(parameters, lr=lr, lr_decay=lr_decay,
                                              initial_accumulator_value=iav, weight_decay=weight_decay)
    elif name == "sparse_adam":
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        lr = kwargs.get('lr', 1e-3)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.SparseAdam = OptimClass(parameters, lr=lr, betas=betas, eps=eps)
    elif name == "rmsprop":
        alpha = kwargs.get('alpha', 0.99)
        eps = kwargs.get('eps', 1e-8)
        weight_decay = kwargs.get('weight_decay', 0)
        momentum = kwargs.get('momentum', 0)
        centered = kwargs.get('centered', False)
        lr = kwargs.get('lr', 1e-2)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.RMSprop = OptimClass(parameters, lr=lr, alpha=alpha, eps=eps,
                                              weight_decay=weight_decay, momentum=momentum, centered=centered)
    elif name == "rprop":
        etas = kwargs.get('etas', (0.5, 1.2))
        step_sizes = kwargs.get('eps', (1e-6, 50))
        lr = kwargs.get('lr', 1e-2)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.Rprop = OptimClass(parameters, lr=lr, etas=etas, step_size=step_sizes)
    elif name == "lbfgs":
        max_iter = kwargs.get('max_iter', 20)
        max_eval = kwargs.get('max_eval', None)
        tolerance_grad = kwargs.get('tolerance_grad', 1e-5)
        tolerance_change = kwargs.get('tolerance_change', 1e-9)
        history_size = kwargs.get('history_size', 100)
        line_search_fn = kwargs.get('line_search_fn', None)
        lr = kwargs.get('lr', 1)
        if type(lr) == list: lr = lr[-1]
        opt_build: optim.LBFGS = OptimClass(parameters, lr=lr, max_iter=max_iter, max_eval=max_eval,
                                            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                                            history_size=history_size, line_search_fn=line_search_fn)
    return opt_build


def string_to_optim(str_optim: str) -> Union[None, optim.Optimizer]:
    map_opt = {
        'sgd': optim.SGD,
        'asgd': optim.ASGD,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'sparse_adam': optim.SparseAdam,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'lbfgs': optim.LBFGS,
    }
    if str_optim in map_opt.keys():
        return map_opt[str_optim]
    else:
        return None


def optim_name(opt: optim.Optimizer):
    if opt == optim.SGD: return "sgd"
    elif opt == optim.ASGD: return "asgd"
    elif opt == optim.Adam: return "adam"
    elif opt == optim.Adamax: return "adamax"
    elif opt == optim.Adadelta: return "adadelta"
    elif opt == optim.Adagrad: return "adagrad"
    elif opt == optim.SparseAdam: return "spars_adam"
    elif opt == optim.RMSprop: return "rmsprop"
    elif opt == optim.Rprop: return "rprop"
    elif opt == optim.LBFGS: return "lbfgs"
    else: return None


if __name__ == '__main__':
    from torchvision import models
    test_model = models.alexnet()
    opt = optim.Adam
    optwr = OptimizerWrapper(test_model)
    myoptim = optwr.create(lr=[0.1], opt='sgd', betas=[0.5, 0.8])
    print(myoptim)








