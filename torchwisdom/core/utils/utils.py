import torch
from torch import Tensor
from typing import *
from itertools import islice


def flatten_check(y_pred: Tensor, y_true: Tensor) -> Tuple:
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    assert len(y_pred) == len(y_true), f"Number element of y_pred and y_true " \
        f"must be the same but got {len(y_pred)} and {len(y_true)}"
    return y_pred, y_true


def flatten_argmax(val_tensor: Tensor) -> Tensor:
    # val_tensor = val_tensor.contiguous().view(-1)
    out = val_tensor.argmax(dim=1)
    return out


def is_flatten_same_dim(y_pred: Tensor, y_true: Tensor) -> bool:
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)
    if len(y_pred) == len(y_true): return True
    else: return False


def chunk(it, size):
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])
