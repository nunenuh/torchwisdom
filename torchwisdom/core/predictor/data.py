from dataclasses import *
from typing import *
import torchvision.transforms as transforms
import torch.nn as nn


__all__ = ['PredictorDataState', 'PredictorModelState', 'PredictorMetaState', 'setattr_parser']


# noinspection PyDataclass
@dataclass
class PredictorDataState:
    input_size: List = field(default_factory=list)
    transform: transforms.Compose = None
    classes: List = field(default_factory=list)
    class_idx: Dict = field(default_factory=dict)
    ctype: str = ''
    dtype: str = ''
    other: Dict = field(default_factory=dict)


@dataclass
class PredictorModelState:
    class_name: str = ''
    class_obj: nn.Module = None
    state_dict: Dict = field(default_factory=dict)
    other: Dict = field(default_factory=dict)


@dataclass
class PredictorMetaState:
    other: Dict = field(default_factory=dict)


def setattr_parser(obj: object, data: Dict):
    for key, val in data.items():
        setattr(obj, key, val)