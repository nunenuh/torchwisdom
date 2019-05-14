from typing import *

import torch

from .data import *


# __all__ = ['Predictor', 'PredictorMetaState', 'PredictorModelState', 'PredictorDataState'
#            '']



class _Predictor(object):
    def __init__(self, file, **kwargs):
        self.file = file
        self.kwargs = kwargs

    def _file_check(self, file): return NotImplementedError()
    def _pre_check(self, data): return NotImplementedError()
    def _pre_predict(self, data): return NotImplementedError()
    def _predict(self, data): return NotImplementedError()
    def _post_check(self, data): return NotImplementedError
    def _post_predict(self, data): return NotImplementedError()
    def predict(self, data): return NotImplementedError()


class Predictor(_Predictor):
    def __init__(self, file, **kwargs):
        super(Predictor, self).__init__(file, **kwargs)
        self.file: Dict = self._file_check(file)
        self.kwargs = kwargs
        self.device = "cpu"
        self._init_dataclass()
        self._init_attribute()

    @staticmethod
    def kwargs_parser(key, **kwargs) -> Any:
        if key in kwargs:
            return kwargs.get(key, None)
        return None

    def _file_check(self, file):
        if type(file) == str:
            map_location = self.kwargs_parser("map_location", **self.kwargs)
            return torch.load(file, map_location=map_location)
        elif type(file) == dict:
            return file
        else:
            raise ValueError("file must be string or dictionary loaded from torch.load!")

    def _init_dataclass(self):
        self.data_state = PredictorDataState()
        self.model_state = PredictorModelState()
        self.meta_state = PredictorMetaState()

    def _init_attribute(self):
        setattr_parser(self.data_state, self.file.get('data', {}))
        setattr_parser(self.model_state, self.file.get("model", {}))
        setattr_parser(self.meta_state, self.file.get("meta", {}))

class VisionSupervisePredictor(Predictor):
    def __init__(self,  file: Union[str, Dict], **kwargs: Any):
        super(VisionSupervisePredictor, self).__init__(file, **kwargs)
        self.model = self.model_state.class_obj
        self.transform = self.data_state.transform


class VisionSemiSupervisePredictor(Predictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(VisionSemiSupervisePredictor, self).__init__(file, **kwargs)


class TextSupervisedPredictor(Predictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(TextSupervisedPredictor, self).__init__(file, **kwargs)
        

class TextUnsupervisedPredictor(Predictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(TextUnsupervisedPredictor, self).__init__(file, **kwargs)
        

class AudioSupervisedPredictor(Predictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(AudioSupervisedPredictor, self).__init__(file, **kwargs)


class AudioUnsupervisedPredictor(Predictor):
    def __init__(self, file: Union[str, Dict], **kwargs: Any):
        super(AudioUnsupervisedPredictor, self).__init__(file, **kwargs)


