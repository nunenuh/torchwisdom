from typing import *
import torch.nn as nn
from torchwisdom.core.statemgr.state import *
from torchwisdom.core.callback import Callback


__all__ = ['ModelCallback', 'ModelCheckPointCallback']


class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.statemgr: StateManager = None

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        model: nn.Module = self.trainer.model
        model_state = self.statemgr.state.get('model')
        model_state['state_dict'] = model.state_dict()
        model_state['classname'] = model.__class__.__name__
        model_state['object'] = model


class ModelCheckPointCallback(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='auto', **kwargs):
        super(ModelCheckPointCallback, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = kwargs.get('verbose', False)
        self.save_best_only = kwargs.get("save_best_only", False)
        self.save_weight_only = kwargs.get("save_best_only", False)
        self.period = kwargs.get("period", 1)
        self.statemgr: StateManager = None

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        model: nn.Module = self.trainer.model

    def on_fit_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_resume_begin(self, *args: Any, **kwargs: Any) -> None:
        model: nn.Module = self.trainer.model

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        metric_state: Dict = self.statemgr.state.get('metric')
        mon_metric = self.monitor.split("_")[1]
        mode = self.get_monitor_mode()
        last_mean = metric_state.get(mode).get(mon_metric).get('mean')[-1]


    def get_monitor_mode(self):
        mode = self.monitor.split("_")[0]
        if mode == 'trn':
            return 'train'
        else:
            return 'valid'

