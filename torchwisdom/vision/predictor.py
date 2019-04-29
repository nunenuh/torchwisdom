from torchwisdom.predictor import VisionSupervisePredictor
from .utils import *
from typing import *
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ConvPredictor']


class ConvPredictor(VisionSupervisePredictor):
    def __init__(self):
        super(ConvPredictor, self).__init__()

    def _pre_check(self, data: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> bool:
        id_data = identify_input(data)
        if id_data is 'string':
            return is_file_pil_compatible(data)
        elif id_data is 'numpy':
            return is_numpy_pil_compatible(data)
        elif id_data is 'pil':
            return is_pil_verified(data)
        elif id_data is 'tensor':
            return is_tensor_image_compatible(data)
        else:
            return False

    def _pre_load(self, data: Union[str, np.ndarray, Image, torch.Tensor]) -> Tuple[Any, str]:
        id_data = identify_input(data)
        if id_data is 'string':
            out = Image.open(data)
        elif id_data is 'numpy':
            out = Image.fromarray(data)
        elif id_data is 'pil' or id_data is 'tensor':
            out = data
        else:
            out = None
        return out, id_data

    def _pre_predict(self, data: Union[str, np.ndarray, Image, torch.Tensor]) -> torch.Tensor:
        feature: torch.Tensor = None
        is_clean = self._pre_check(data)
        if is_clean:
            loaded_data, loaded_type = self._pre_load(data)
            if loaded_data and loaded_type is not 'tensor':
                feature: torch.Tensor = self.transform(loaded_data)
                feature = feature.unsqueeze(dim=0)
            else:
                if is_tensor_single_image(data):
                    feature = data.unsqueeze(dim=0)
        return feature

    def _predict(self, feature: Union[str, np.ndarray, Image, torch.Tensor]):
        prediction = None
        if feature:
            feature.to(self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            prediction = self.model(feature)
        return prediction

    def _post_predict(self, prediction: torch.Tensor):
        is_label = is_tensor_label(prediction)
        if is_label:
            if is_tensor_label_batch(prediction):
                torch.argmax(prediction, dim=1)
            elif is_tensor_label_single(prediction):
                ...
            else:
                ...

    def _post_check(self, data):
        ...

    def _topk(self, data, k_val=5):
        feature = self._pre_predict(data)
        prediction = self._predict(feature)
        if is_tensor_label(prediction):
            output = F.log_softmax(prediction, dim=1)
            ps = torch.exp(output)
            probability, classes = ps.topk(k_val, dim=1, largest=True, sorted=True)
            return probability, classes
        return False

    def predict(self, data):
        feature = self._pre_predict(data)
        prediction = self._predict(feature)
        information = self._post_predict(prediction)
        return information


if __name__ == "__main__":
    ...

