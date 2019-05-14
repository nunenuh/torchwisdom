import torch
from typing import *

class ExporterBuilder(object):
    def __init__(self, trainer):
        self.trainer  = trainer
        self.data = self.trainer.data
        self.state_manager = self.trainer.state_manager
        self.state = {}

    def export(self, path: str, **kwargs: Any):
        self._prepare_export()
        torch.save(self.state, path)

    def _prepare_export(self):
        self._build_data_state()
        self._build_model_state()
        self._build_meta_state()

    def _build_data_state(self):
        data = {
            'ctype': self.trainer.data.case_type,
            'dtype': self.trainer.data.data_type,
            'classes': self.trainer.data.trainset_attr.classes,
            'class_idx': self.trainer.data.trainset_attr.class_to_idx,
            'transform': self.trainer.data.validset_attr.transform
        }
        self.state.update({'data': data})

    def _build_model_state(self):
        model: Dict = self.state_manager.state.get("model")
        self.state.update({'model': model})

    def _build_meta_state(self):
        meta_info = {

        }
        self.state.update({'meta': meta_info})

    def build_state(self):
        self._prepare_export()




