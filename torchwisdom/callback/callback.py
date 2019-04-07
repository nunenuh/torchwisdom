from typing import *
from dataclasses import dataclass
# from torchwisdom.trainer import Trainer

_TRAIN, _VALID, _TEST, _PREDICT = 'train', 'valid', 'test', 'predict'
_FORWARD, _BACKWARD = 'forward', 'backward'


class Hook(object):

    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_epoch_resume(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_train_begin(self, *args: Any, **kwargs: Any)->None:
        pass

    def on_train_batch_begin(self, *args: Any, **kwargs: Any)->None:
        pass

    def on_train_batch_end(self, *args: Any, **kwargs: Any)->None:
        pass

    def on_train_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_valid_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_valid_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_valid_batch_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_valid_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_predict_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_predict_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_test_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_test_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_forward_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_forward_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_backward_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_backward_end(self, *args: Any, **kwargs: Any) -> None:
        pass


class Callback(Hook):
    """ Base for all callback class """
    trainer: object = None

    def __init__(self):
        pass

    def set_trainer(self, trainer):
        self.trainer = trainer



@dataclass
class CallbackHandler(object):
    trainer = None
    callbacks: Collection[Callback] = None
    metrics: Collection[Callback] = None

    def __init__(self, trainer = None,  metrics: Collection[Callback] = None, callbacks: Collection[callbacks] = None):
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        self._build_callbacks()
        if trainer:
            self.set_trainer(trainer)

    def _build_callbacks(self):
        self.callbacks = [callback for callback in self.callbacks]
        self.metrics = [metric for metric in self.metrics]

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)
        for metric in self.metrics:
            metric.set_trainer(trainer)

    def _call_batch_hook(self, mode, hook):
        hook_name = f'on_{mode}_batch_{hook}'

        for callback in self.callbacks:
            batch_hook = getattr(callback, hook_name)
            batch_hook() # I still confuse in here

        for metric in self.metrics:
            batch_hook = getattr(metric, hook_name)
            batch_hook() # I still confuse in here

    def _call_begin_hook(self, mode):
        if mode == _TRAIN:
            self.on_train_begin()
        elif mode == _VALID:
            self.on_valid_begin()
        elif mode == _TEST:
            self.on_test_begin()
        elif mode == _PREDICT:
            self.on_predict_begin()

    def _call_end_hook(self, mode):
        if mode == _TRAIN:
            self.on_train_end()
        elif mode == _VALID:
            self.on_valid_end()
        elif mode == _TEST:
            self.on_test_end()
        elif mode == _PREDICT:
            self.on_predict_end()

    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin()
        for metric in self.metrics:
            metric.on_epoch_begin()

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end()
        for metric in self.metrics:
            metric.on_epoch_end()

    def on_epoch_resume(self, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            callback.on_epoch_resume()
        for metric in self.metrics:
            metric.on_epoch_resume()

    def on_train_begin(self, *args: Any, **kwargs: Any)->None:
        self._call_begin_hook(_TRAIN)

    def on_train_batch_begin(self, *args: Any, **kwargs: Any)->None:
        self._call_batch_hook(_TRAIN, 'begin')

    def on_train_batch_end(self, *args: Any, **kwargs: Any)->None:
        self._call_batch_hook(_TRAIN, 'end')

    def on_train_end(self, *args: Any, **kwargs: Any) -> None:
        self._call_end_hook(_TRAIN)

    def on_valid_begin(self, *args: Any, **kwargs: Any) -> None:
        self._call_begin_hook(_VALID)

    def on_valid_batch_begin(self, *args: Any, **kwargs: Any) -> None:
        self._call_batch_hook(_VALID, 'begin')

    def on_valid_batch_end(self, *args: Any, **kwargs: Any) -> None:
        self._call_batch_hook(_VALID, 'end')

    def on_valid_end(self, *args: Any, **kwargs: Any) -> None:
        self._call_end_hook(_TRAIN)

    def on_predict_begin(self, *args: Any, **kwargs: Any) -> None:
        self._call_begin_hook(_PREDICT)

    def on_predict_end(self, *args: Any, **kwargs: Any) -> None:
        self._call_end_hook(_PREDICT)

    def on_test_begin(self, *args: Any, **kwargs: Any) -> None:
        self._call_begin_hook(_TEST)

    def on_test_end(self, *args: Any, **kwargs: Any) -> None:
        self._call_end_hook(_TEST)

    def on_forward_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_forward_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_backward_begin(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_backward_end(self, *args: Any, **kwargs: Any) -> None:
        pass





if __name__ == '__main__':
    # cb_handler = CallbackHandler()
    cb = Callback()
    method = getattr(cb, 'on_epoch_begin')
    method(10, test='10')