from typing import *
import torch


__all__ = []


def idx_to_class_mapper(class_to_idx: Dict):
    out = {}
    for k, v in class_to_idx.items():
        out.update({v:k})
    return out


def idx_to_class(idx_data: torch.Tensor, classes: List):
    if idx_data.dim() == 1:
        label = []
        for idx in idx_data:
            label.append(classes[idx])
        return label
    else:
        raise ValueError(f"Tensor Dimension must be one, but got {idx_data.dim()} dimension, "
                         f"are you sure you have argmax the Tensor before you fill "
                         f"to parameter of this function?")


if __name__ == '__main__':
    pass
