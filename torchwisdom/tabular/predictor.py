from typing import *

import pandas as pd
import torch
import torch.nn.functional as F

from ..core.predictor import Predictor
from .utils import *
import numpy as np

__all__ = ['TabularSupervisedPredictor', 'TabularUnsupervisedPredictor',
           'TabularClassifierPredictor', 'TabularRegressorPredictor']


class TabularSupervisedPredictor(Predictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(TabularSupervisedPredictor, self).__init__(file, **kwargs)
        self.model = self.model_state.class_obj
        self.transform = self.data_state.transform

    def _pre_check(self, feature: Any = None):
        # print(feature.dim())
        pass

    def _pre_predict(self, feature: Union[str, pd.DataFrame, torch.Tensor, np.ndarray, List] = None,
                     use_transform=True) -> torch.Tensor:
        self._pre_check(feature)
        if type(feature) == str:
            frame = pd.read_csv(feature)
            data = frame.values
        elif type(feature) == pd.DataFrame:
            data = feature.values
        elif type(feature) == torch.Tensor:
            data = feature.numpy()
            if feature.dim() < 2:
                data = feature.unsqueeze(dim=0)
                data = data.numpy()
        elif type(feature) == np.ndarray:
            data = feature
        elif type(feature) == list:
            data = np.array([feature])
        else:
            raise TypeError("Incompatible type for continue processing to model!")
        if use_transform:
            data = self.transform(data)
        return data


class TabularUnsupervisedPredictor(Predictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(TabularUnsupervisedPredictor, self).__init__(file, **kwargs)


class TabularClassifierPredictor(TabularSupervisedPredictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(TabularClassifierPredictor, self).__init__(file, **kwargs)

    def _predict(self, feature: torch.Tensor):
        prediction = None
        if len(feature) > 0:
            feature = feature.to(self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(feature)
        return prediction

    def _post_predict(self, prediction: torch.Tensor, use_topk: bool = False, kval: int = 5) -> Union[Any, Tuple]:
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

    @staticmethod
    def _build_topk_series(predict, data_dict, target_columns, kval):
        for i in range(kval):
            percent = predict[0]
            classes = predict[2]
            data = []
            for cls, prc in zip(classes, percent):
                if len(cls) == 1:
                    data.append(f"{cls[0]} ({prc[0] * 100:.4f}%)")
                else:
                    data.append(f"{cls[i]} ({prc[i] * 100:.4f}%)")
            data_dict.update({target_columns[0] + "_predict_top" + str(i + 1): data})

    def _show_as_dataframe(self, feature: torch.Tensor, predict: torch.Tensor, target: torch.Tensor, kval,
                           feature_columns=[], target_columns=[]) -> pd.DataFrame:
        if target.dim() < 2:
            tdn = 2 - target.dim()
            for i in range(tdn):
                target = target.unsqueeze(dim=0)
            target = self._class_label(target[0])

        data_dict = {}
        for idx, col in enumerate(feature_columns):
            data_dict.update({col: feature[:, idx]})

        if target is not None:
            data_dict.update({target_columns[0] + "_truth": target[:]})

        if len(predict) == 3:
            self._build_topk_series(predict, data_dict, target_columns, kval)
        elif len(predict) == 2:
            data_dict.update({target_columns[0] + "_predict": predict[1]})

        df = pd.DataFrame(data_dict)
        # df = data_dict
        return df

    def predict(self, feature: Union[str, pd.DataFrame, torch.Tensor, np.ndarray, List] = None,
                target: Any = None, transform=None, use_topk: bool = False, kval: int = 1,
                show_table: bool = False):
        if transform: self.transform = transform
        feat = self._pre_predict(feature)
        prediction = self._predict(feat)
        result = self._post_predict(prediction, use_topk=use_topk, kval=kval)
        if show_table:
            feat = self._pre_predict(feature, use_transform=False)
            return self._show_as_dataframe(feat, result, target, kval=kval,
                                           feature_columns=self.data_state.feature_columns,
                                           target_columns=self.data_state.target_columns)
        return result


class TabularRegressorPredictor(TabularSupervisedPredictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(TabularRegressorPredictor, self).__init__(file, **kwargs)

    def _predict(self, feature: torch.Tensor):
        prediction = None
        if len(feature):
            feature.to(self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(feature)
        return prediction

    def _post_predict(self, prediction: torch.Tensor) -> Union[None, Tuple]:
        is_clean = self._post_check(prediction)
        if is_clean:
            return prediction
        return None

    def _post_check(self, prediction: torch.Tensor) -> bool:
        # check output is clean (classfier label image, not image)
        return True

    @staticmethod
    def _show_as_dataframe(feature: torch.Tensor, predict: torch.Tensor, target: torch.Tensor,
                           feature_columns=[], target_columns=[]) -> pd.DataFrame:

        # print(len(feature), len(target))
        # if target.dim() < 2:
        #     tdn = 2 - target.dim()
        #     for i in range(tdn):
        #         target = target.unsqueeze(dim=0)

        data_dict = {}
        for idx, col in enumerate(feature_columns):
            data_dict.update({col: feature[:, idx]})

        data_dict.update({target_columns[0] + "_predict": predict[:, 0]})
        if target is not None:
            data_dict.update({target_columns[0] + "_truth": target.squeeze()})

        # print(len(feature), len(predict), len(target))

        df = pd.DataFrame(data_dict)
        # df = data_dict
        return df

    def predict(self, feature, target=None, show_table=False, transform=None):
        if transform: self.transform = transform
        feat = self._pre_predict(feature)
        prediction = self._predict(feat)
        result = self._post_predict(prediction)
        if show_table:
            feat = self._pre_predict(feature, use_transform=False)
            return self._show_as_dataframe(feat, result, target,
                                           feature_columns=self.data_state.feature_columns,
                                           target_columns=self.data_state.target_columns)
        else:
            return result


if __name__ == "__main__":
    ...
