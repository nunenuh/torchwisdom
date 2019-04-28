from pathlib import Path
from PIL import Image
import numpy as np
import torch
from typing import *
import PIL
import PIL.JpegImagePlugin


__all__ = ['is_file_pil_compatible', 'is_numpy_pil_compatible',
           'is_tensor_single_image', 'is_tensor_batch_image', 'is_pil_verified', 'is_tensor_image_compatible']


def is_file_pil_compatible(path: str) -> bool:
    p = Path(path)
    if p.exists() and p.is_file():
        try:
            Image.open(str(p))
            return True
        except:
            return False
    else:
        return False


def is_numpy_pil_compatible(np_array: np.ndarray) -> bool:
    try:
        Image.fromarray(np_array)
        return True
    except:
        return False


def is_tensor_image_compatible(tensor_data: torch.Tensor) -> bool:
    if tensor_data.dim() == 4:
        return is_tensor_batch_image(tensor_data)
    elif tensor_data.dim() == 3:
        return is_tensor_single_image(tensor_data)
    else:
        return False


def is_tensor_batch_image(tensor_data: torch.Tensor) -> bool:
    if tensor_data.dim() == 4:
        if 1 <= tensor_data.size()[1] <= 3:
            return True
        else:
            return False
    else:
        return False


def is_tensor_single_image(tensor_data: torch.Tensor) -> bool:
    if tensor_data.dim() == 3:
        if 1 <= tensor_data.size()[0] <= 3:
            return True
        else:
            return False
    else:
        return False


def is_pil_verified(data: Image.Image) -> bool:
    try:
        data.verify()
        return True
    except:
        return False


def identify_data(data: Union[str, np.ndarray, torch.Tensor, Image]):
    if type(data) == str:
        return 'string'
    elif type(data) == np.ndarray:
        return 'numpy'
    elif type(data) == torch.Tensor:
        return 'tensor'
    elif isinstance(data, Image.Image):
        return "pil"
    else:
        return None


