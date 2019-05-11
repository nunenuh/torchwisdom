from torchwisdom.core.predictor import _Predictor
from .utils import *
from typing import *
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchwisdom.core.utils import DatasetCollector
import pandas as pd


__all__ = ['TabularClassifierPredictor']


class TabularSupervisedPredictor(_Predictor):
    def __init__(self, model: nn.Module, data: DatasetCollector, transform=None):
        super(TabularSupervisedPredictor, self).__init__(model, data, transform)

    def _pre_check(self, *args: Any, **kwargs: Any) -> bool:
        la, lk = len(args), len(kwargs)
        if la == 0 and lk == 0:
            return False
        elif la > 0 and lk > 0:
            return False
        else:
            return True

    def _pre_predict(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        is_clean = self._pre_check(*args, **kwargs)
        if is_clean:
            if len(args) > 0 and len(kwargs) == 0:
                data = torch.Tensor([args])
            else:
                if 'csv_file' in kwargs:
                    csv_file = kwargs.get('csv_file')
                    frame = pd.read_csv(csv_file)
                    data = torch.from_numpy(frame.values).float()
                elif 'dataframe' in kwargs:
                    frame = kwargs.get('dataframe')
                    data = torch.from_numpy(frame.values).float()
                elif 'tensor_data' in kwargs:
                    data = kwargs.get('tensor_data')
                    if data.dim() <2:
                        data = data.unsqueeze(dim=0)
                elif 'numpy_data' in kwargs:
                    numpy_data = kwargs.get('numpy_data')
                    data = torch.from_numpy(numpy_data).float()
                elif 'list_data' in kwargs:
                    list_data = kwargs.get("list_data")
                    data = torch.Tensor([list_data]).float()
                else:
                    data = None
        else:
            data = None

        return data

    @staticmethod
    def _clean_remove_kwargs(key, **kwargs):
        if key in kwargs: kwargs.pop(key)
        return kwargs

    def _clean_kwargs(self, **kwargs: Any) -> Any:
        kwargs = self._clean_remove_kwargs('use_topk', **kwargs)
        kwargs = self._clean_remove_kwargs('kval', **kwargs)
        kwargs = self._clean_remove_kwargs('target', **kwargs)
        kwargs = self._clean_remove_kwargs('show_table', **kwargs)
        kwargs = self._clean_remove_kwargs('feature_columns', **kwargs)
        kwargs = self._clean_remove_kwargs('target_columns', **kwargs)
        return kwargs


class TabularUnsupervisedPredictor(_Predictor):
    def __init__(self):
        super(TabularUnsupervisedPredictor, self).__init__()


class TabularClassifierPredictor(TabularSupervisedPredictor):
    def __init__(self, model: nn.Module, data: DatasetCollector, transform=None):
        super(TabularClassifierPredictor, self).__init__(model, data, transform)
        if transform is None:
            self.transform = self.data.validset_attr.transform

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

    @staticmethod
    def _build_topk_series( predict, data_dict, target_columns, kval):
        for i in range(kval):
            percent = predict[0]
            classes = predict[2]
            data = []
            for cls, prc in zip(classes, percent):
                if len(cls)==1:
                    data.append(f"{cls[0]} ({prc[0]*100:.4f}%)")
                else:
                    data.append(f"{cls[i]} ({prc[i]*100:.4f}%)")
            data_dict.update({target_columns[0] + "_predict_top" + str(i + 1): data})

    def _show_as_dataframe(self, feature: torch.Tensor, predict: torch.Tensor,
                           target: torch.Tensor, kval,  **kwargs: Any) -> pd.DataFrame:
        feature_columns = kwargs.get("feature_columns", [])
        target_columns = kwargs.get("target_columns", [])
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
        return df

    def predict(self, *args: Any, **kwargs: Any):
        use_topk, kval = kwargs.get("use_topk", False), kwargs.get("kval", 1)
        target = kwargs.get("target", None)
        show_table = kwargs.get("show_table", False)
        feature_columns = kwargs.get("feature_columns", None)
        target_columns = kwargs.get("target_columns", None)
        kwargs = self._clean_kwargs(**kwargs)

        feature = self._pre_predict(*args, **kwargs)
        prediction = self._predict(feature)
        result = self._post_predict(prediction, use_topk=use_topk, kval=kval)
        if show_table:
            return self._show_as_dataframe(feature, result, target,
                                           feature_columns=feature_columns,
                                           target_columns=target_columns,
                                           kval=kval)
        else:
            return result


class TabularRegressorPredictor(TabularSupervisedPredictor):
    def __init__(self, model: nn.Module, data: DatasetCollector, transform=None):
        super(TabularRegressorPredictor, self).__init__(model, data, transform)
        if transform is None:
            self.transform = self.data.validset_attr.transform

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
    def _show_as_dataframe(feature: torch.Tensor, predict: torch.Tensor,
                           target: torch.Tensor, **kwargs: Any) -> pd.DataFrame:
        feature_columns = kwargs.get("feature_columns", [])
        target_columns = kwargs.get("target_columns", [])
        if target.dim() < 2:
            tdn = 2 - target.dim()
            for i in range(tdn):
                target = target.unsqueeze(dim=0)

        data_dict = {}
        for idx, col in enumerate(feature_columns):
            data_dict.update({col: feature[:, idx]})

        data_dict.update({target_columns[0] + "_predict": predict[:, 0]})
        if target is not None:
            data_dict.update({target_columns[0] + "_truth": target[:, 0]})

        df = pd.DataFrame(data_dict)
        return df

    def predict(self, *args: Any, **kwargs: Any):
        target = kwargs.get("target", None)
        show_table = kwargs.get("show_table", False)
        feature_columns = kwargs.get("feature_columns", None)
        target_columns = kwargs.get("target_columns", None)
        kwargs = self._clean_kwargs(**kwargs)

        feature = self._pre_predict(*args, **kwargs)
        prediction = self._predict(feature)
        result = self._post_predict(prediction)
        if show_table:
            return self._show_as_dataframe(feature, result, target,
                                           feature_columns=feature_columns,
                                           target_columns=target_columns)
        else:
            return result


if __name__ == "__main__":
    ...


