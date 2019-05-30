from .utils import *
from typing import *
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.data import DataCapsule
from ..core.predictor import VisionSupervisePredictor, VisionSemiSupervisePredictor
from .viz import *

__all__ = ['ConvClassifierPredictor', 'ConvAutoEncoderPredictor']


class ConvClassifierPredictor(VisionSupervisePredictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(ConvClassifierPredictor, self).__init__(file, **kwargs)
        self.model = self.model_state.class_obj
        self.transform = self.data_state.transform

    def _pre_check(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> bool:
        id_data = identify_input(feature)
        if id_data is 'string':
            return is_file_pil_compatible(feature)
        elif id_data is 'numpy':
            return is_numpy_pil_compatible(feature)
        elif id_data is 'pil':
            return is_pil_verified(feature)
        elif id_data is 'tensor':
            return is_tensor_image_compatible(feature)
        elif id_data is 'list':
            return is_list_compatible(feature)
        else:
            return False

    def _pre_load(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Tuple[Any, str]:
        id_data = identify_input(feature)
        if id_data == 'string':
            out = Image.open(feature)
            out = out.convert("RGB")
        elif id_data == 'numpy':
            out = Image.fromarray(feature)
            out = out.convert("RGB")
        elif id_data == 'pil' or id_data == 'tensor':
            out = feature
        elif id_data == 'list':
            ar = []
            for feat in feature:
                feat = feat.convert("RGB")
                feat = np.array(feat)
                ar.append(feat)
            ar = np.array(ar)
            out = torch.from_numpy(ar)
            out = out.permute(0, 3, 1, 2).float()
            # print(out.shape)
        else:
            out = None
        return out, id_data

    def _pre_predict(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        feat: torch.Tensor = torch.Tensor()
        is_clean = self._pre_check(feature)
        if is_clean:
            loaded_data, loaded_type = self._pre_load(feature)
            if type(loaded_data) != type(None):
                if loaded_type == 'tensor':
                    feat: torch.Tensor = loaded_data
                    if not is_tensor_batch_image(feature):
                        feat = feature.unsqueeze(dim=0)
                elif loaded_type == 'list':
                    feat = loaded_data
                else:
                    feat: torch.Tensor = self.transform(loaded_data)
                    feat = feat.unsqueeze(dim=0)
            else:
                if is_tensor_single_image(feature):
                    feat = feature.unsqueeze(dim=0)
        return feat

    def _predict(self, feature: torch.Tensor):
        prediction = None
        if len(feature):
            feature = feature.to(self.device)
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
        classes = self.data_state.classes
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

    def _show_images(self, feature, result, target):
        title = []
        for idx, info in enumerate(zip(result[0], result[1], target)):
            prob_pred = info[0] * 100
            class_pred = self.data_state.classes[info[1]]
            # print(info)
            if type(info[2]) != str:
                class_targ = self.data_state.classes[info[2]]
            else:
                class_targ = info[2]
            t = f'{class_pred}({prob_pred.item():.2f}%) / {class_targ}'
            title.append(t)

        # print(feature)

        images = feature.permute(0, 2, 3, 1)
        show_figure(images, title, figsize=(15, 15))

    def predict(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor],
                target=None, use_topk=True,
                kval=1, show_images=False):
        feat = self._pre_predict(feature)
        prediction = self._predict(feat)
        result = self._post_predict(prediction, use_topk=use_topk, kval=kval)
        if show_images:
            self._show_images(feat, result, target)
        else:
            return result


class ConvAutoEncoderPredictor(VisionSemiSupervisePredictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(ConvAutoEncoderPredictor, self).__init__(file, **kwargs)
        self.model = self.model_state.class_obj

    def _pre_check(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> bool:
        id_data = identify_input(feature)
        if id_data is 'string':
            return is_file_pil_compatible(feature)
        elif id_data is 'numpy':
            return is_numpy_pil_compatible(feature)
        elif id_data is 'pil':
            return is_pil_verified(feature)
        elif id_data is 'tensor':
            return is_tensor_image_compatible(feature)
        elif id_data is 'list':
            return is_list_compatible(feature)
        else:
            return False

    def _pre_load(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> Tuple[Any, str]:
        id_data = identify_input(feature)
        if id_data == 'string':
            out = Image.open(feature)
            out = out.convert("RGB")
        elif id_data == 'numpy':
            out = Image.fromarray(feature)
            out = out.convert("RGB")
        elif id_data == 'pil' or id_data == 'tensor':
            out = feature
        elif id_data == 'list':
            ar = []
            for feat in feature:
                feat = feat.convert("RGB")
                feat = np.array(feat)
                ar.append(feat)
            out = torch.from_numpy(ar)
        else:
            out = None
        return out, id_data

    def _pre_predict(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        feat: torch.Tensor = torch.Tensor()
        is_clean = self._pre_check(feature)
        if is_clean:
            loaded_data, loaded_type = self._pre_load(feature)
            if type(loaded_data) != type(None):
                if loaded_type == 'tensor':
                    feat: torch.Tensor = loaded_data
                    if not is_tensor_batch_image(feature):
                        feat = feature.unsqueeze(dim=0)
                else:
                    feat: torch.Tensor = self.transform(loaded_data)
                    feat = feat.unsqueeze(dim=0)
            else:
                if is_tensor_single_image(feature):
                    feat = feature.unsqueeze(dim=0)
        return feat

    def _predict(self, feature: torch.Tensor):
        prediction = None
        if len(feature):
            feature = feature.to(self.device)
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

    def predict(self, feature: Union[str, np.ndarray, Image.Image, torch.Tensor]):
        feat = self._pre_predict(feature)
        prediction = self._predict(feat)
        result = self._post_predict(prediction)
        return result

class ConvSiamesePredictor(VisionSemiSupervisePredictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(ConvSiamesePredictor, self).__init__(file, **kwargs)
        self.model = self.model_state.class_obj

if __name__ == "__main__":
    ...
