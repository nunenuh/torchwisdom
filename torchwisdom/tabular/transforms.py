import torch
import numpy as np
from typing import *
from ..core.nn import functional as N

__all__ = ['NumpyToTensor', 'ListToNumpy', 'ToFloatTensor', 'ToLongTensor']


class ListToNumpy(object):
    def __init__(self):
        super(ListToNumpy, self).__init__()

    def __call__(self, list_data: List):
        return np.array(list_data)


class NumpyToTensor(object):
    def __init__(self):
        super(NumpyToTensor, self).__init__()

    def __call__(self, ndarray: np.ndarray):
        return torch.from_numpy(ndarray)


class ToFloatTensor(object):
    def __init__(self):
        super(ToFloatTensor, self).__init__()

    def __call__(self, x: torch.Tensor):
        return x.float()


class ToLongTensor(object):
    def __init__(self):
        super(ToLongTensor, self).__init__()

    def __call__(self, x: torch.Tensor):
        return x.long()


class Normalize(object):
    def __init__(self, feature_columns: List, feature_stats: Dict, mode='minmax'):
        self.feature_columns = feature_columns
        self.feature_stats = feature_stats
        self.mode = mode

    def __call__(self, feature: torch.Tensor):
        feat = feature.clone()
        # result = []
        # idx: int = 0
        for idx, col in enumerate(self.feature_columns):
            # print(idx, col)
            stats = self.feature_stats[col]
            # print(stats)
            if self.mode == 'minmax':
                if feat.dim() == 1:
                    feat[idx] = N.normalization(feat[idx], stats['min'], stats['max'])
                else:
                    feat[:, idx] = N.normalization(feat[:, idx], stats['min'], stats['max'])
            else:
                if feat.dim() == 1:
                    feat[idx] = N.standardization(feat[idx], stats['mean'], stats['std'])
                else:
                    feat[:, idx] = N.standardization(feat[:, idx], stats['mean'], stats['std'])
        return feat.float()
