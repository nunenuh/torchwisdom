from torchwisdom.core.predictor import VisionSupervisePredictor, VisionSemiSupervisePredictor
from .utils import *
from typing import *
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchwisdom.core.utils import DatasetCollector


__all__ = ['ConvClassifierPredictor', 'ConvAutoEncoderPredictor']


class ConvClassifierPredictor(VisionSupervisePredictor):
    def __init__(self, model: nn.Module, data: DatasetCollector, transform=None):
        super(ConvClassifierPredictor, self).__init__(model, data, transform)
        if transform is None:
            self.transform = self.data.validset_attr.transform

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

    def _pre_load(self, data: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Tuple[Any, str]:
        id_data = identify_input(data)
        if id_data == 'string':
            out = Image.open(data)
            out = out.convert("RGB")
        elif id_data == 'numpy':
            out = Image.fromarray(data)
            out = out.convert("RGB")
        elif id_data == 'pil' or id_data == 'tensor':
            out = data
        else:
            out = None
        return out, id_data

    def _pre_predict(self, data: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        feature: torch.Tensor = torch.Tensor()
        is_clean = self._pre_check(data)
        if is_clean:
            loaded_data, loaded_type = self._pre_load(data)
            if type(loaded_data) != type(None):
                if loaded_type == 'tensor':
                    feature: torch.Tensor = loaded_data
                    if not is_tensor_batch_image(feature):
                        feature = feature.unsqueeze(dim=0)
                else:
                    feature: torch.Tensor = self.transform(loaded_data)
                    feature = feature.unsqueeze(dim=0)
            else:
                if is_tensor_single_image(data):
                    feature = data.unsqueeze(dim=0)
        return feature

    def _predict(self, feature: torch.Tensor):
        prediction = None
        if len(feature):
            feature.to(self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(feature)
        return prediction

    def _post_predict(self, prediction: torch.Tensor, use_topk: bool = False, kval: int = 5) -> Union[None, Tuple]:
        is_clean = self._post_check(prediction)
        if is_clean:
            if not use_topk:
                return self._predict_label(prediction)
            else:
                return self._predict_topk(prediction, kval=kval)
        return None

    def _post_check(self, prediction: torch.Tensor) -> bool:
        # check output is clean (classfier label image, not image)
        return True

    def _class_label(self, class_index: torch.Tensor, is_topk=False) -> Union[str, List, None]:
        class_label = []
        classes = self.data.trainset_attr.classes
        if classes:
            if not is_topk:
                if len(class_index) >= 2:
                    for cidx in class_index:
                        class_label.append(classes[cidx])
                    return class_label
                else:
                    return classes[class_index]
            else:
                for ctn in class_index:
                    topk_label = []
                    for cidx in ctn:
                        topk_label.append(classes[cidx])
                    class_label.append(topk_label)
                return class_label
        return None

    def _predict_topk(self, prediction: torch.Tensor, kval: int = 5) -> Union[bool, Tuple]:
        if is_tensor_label(prediction):
            output = F.log_softmax(prediction, dim=1)
            ps = torch.exp(output)
            probability, class_index = ps.topk(kval, dim=1, largest=True, sorted=True)
            class_label = self._class_label(class_index, is_topk=True)
            return probability, class_index, class_label
        return False

    def _predict_label(self, prediction: torch.Tensor) -> Union[bool, Tuple]:
        if is_tensor_label(prediction):
            class_index = torch.argmax(prediction, dim=1)
            class_label = self._class_label(class_index)
            return class_index, class_label
        return False

    def predict(self, data: Union[str, np.ndarray, Image.Image, torch.Tensor], use_topk=False, kval=5):
        feature = self._pre_predict(data)
        prediction = self._predict(feature)
        result = self._post_predict(prediction, use_topk=use_topk, kval=kval)
        return result


class ConvAutoEncoderPredictor(VisionSemiSupervisePredictor):
    def __init__(self, model: nn.Module, data: DatasetCollector, transform=None):
        super(ConvAutoEncoderPredictor, self).__init__(model, data, transform)
        if transform is None:
            self.transform = self.data.validset_attr.transform

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

    def _pre_load(self, data: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Tuple[Any, str]:
        id_data = identify_input(data)
        if id_data == 'string':
            out = Image.open(data)
            out = out.convert("RGB")
        elif id_data == 'numpy':
            out = Image.fromarray(data)
            out = out.convert("RGB")
        elif id_data == 'pil' or id_data == 'tensor':
            out = data
        else:
            out = None
        return out, id_data

    def _pre_predict(self, data: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        feature: torch.Tensor = torch.Tensor()
        is_clean = self._pre_check(data)
        if is_clean:
            loaded_data, loaded_type = self._pre_load(data)
            if type(loaded_data) != type(None):
                if loaded_type == 'tensor':
                    feature: torch.Tensor = loaded_data
                    if not is_tensor_batch_image(feature):
                        feature = feature.unsqueeze(dim=0)
                else:
                    feature: torch.Tensor = self.transform(loaded_data)
                    feature = feature.unsqueeze(dim=0)
            else:
                if is_tensor_single_image(data):
                    feature = data.unsqueeze(dim=0)
        return feature

    def _predict(self, feature: torch.Tensor):
        prediction = None
        if len(feature):
            feature.to(self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(feature)
        return prediction

    def _post_predict(self, prediction: torch.Tensor, use_topk: bool = False, kval: int = 5) -> Union[None, Tuple]:
        is_clean = self._post_check(prediction)
        if is_clean:
            return prediction
        return None

    def _post_check(self, prediction: torch.Tensor) -> bool:
        # check output is clean (classfier label image, not image)
        return True


    def predict(self, data: Union[str, np.ndarray, Image.Image, torch.Tensor]):
        feature = self._pre_predict(data)
        prediction = self._predict(feature)
        result = self._post_predict(prediction)
        return result



if __name__ == "__main__":
    ...

