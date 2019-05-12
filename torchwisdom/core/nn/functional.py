import torch
import pandas as pd
import numpy as np
from typing import *


__all__ = ['normalization','standardization']


def normalization(x: Any, xmin: Any, xmax: Any) -> Any:
    return (x - xmin) / (xmax - xmin)


def standardization(x: Any, mean: float, std: float) -> Any:
    return (x - mean) / std





