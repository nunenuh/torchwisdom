from typing import *
from dataclasses import dataclass
from collections import OrderedDict


__all__ = ['Hook', 'Callback', 'CallbackHandler']


_TRAIN, _VALIDATE, _TEST, _PREDICT = 'train', 'validate', 'test', 'predict'
_BATCH, _FORWARD, _BACKWARD = 'batch', 'forward', 'backward'
_FIT, _RESUME, _EPOCH = 'fit', 'resume', 'epoch'
_BEGIN, _END = 'begin', 'end'


class Hook(object):
    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_fit_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_resume_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_resume_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_batch_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_forward_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_forward_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_backward_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_backward_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_train_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_validate_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_validate_batch_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_validate_batch_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_validate_forward_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_validate_forward_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_validate_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_predict_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_predict_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_test_begin(self, *args: Any, **kwargs: Any) -> None: pass
    def on_test_end(self, *args: Any, **kwargs: Any) -> None: pass


class Callback(Hook):
    """ Base for all callbacks class """

    def __init__(self, trainer=None):
        super(Callback, self).__init__()
        self.trainer: object = trainer
        self.handler: object = None
        self.statemgr: object = None
        self.optimizer: object = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_statemgr(self, statemgr):
        self.statemgr = statemgr

    def set_handler(self, handler):
        self.handler = handler


@dataclass
class CallbackHandler(object):
    trainer: object = None
    callbacks: List[Callback] = None

    def __init__(self, trainer, callbacks: List[Callback] = None):
        self.trainer = trainer
        self.callbacks = callbacks or []
        self.callbacks_odict = OrderedDict()
        # self._build_callbacks()

    def _add_odict(self, callback):
        name = callback.__class__.__name__
        obj = callback
        dc = {name: obj}
        self.callbacks_odict.update(dc)

    def add(self, callbacks: Union[List[Callback], Callback]):
        if type(callbacks) == list:
            for callback in callbacks:
                callback.set_trainer(self.trainer)
                callback.set_handler(self)
                callback.set_statemgr(self.trainer.state_manager)
                callback.set_optimizer(self.trainer.optimizer)
                self.callbacks.append(callback)
                self._add_odict(callback)
        else:
            callbacks.set_trainer(self.trainer)
            callbacks.set_statemgr(self.trainer.state_manager)
            callbacks.set_optimizer(self.trainer.optimizer)
            callbacks.set_handler(self)
            self.callbacks.append(callbacks)
            self._add_odict(callbacks)

    def is_exist(self, callback):
        status = False
        for cb in self.callbacks:
            if cb.__class__.__name__ == callback.__class__.__name__:
                status = True
        return status

    def reconnect_callback(self):
        """
        Reconnect all callbacks in the list or self.callbacks
        :return:
        """

        for callback in self.callbacks:
            callback.set_trainer(self.trainer)
            callback.set_optimizer(self.trainer.optimizer)
            callback.set_handler(self)
            callback.set_statemgr(self.trainer.state_manager)

    def rearrange_callback(self):
        """
        This method is for rearrange callback position, StateManagerCallback must be the first callback in list.
        The purpose of this method is to make sure that StateManagerCallback has to be the first callback to execute
        :return:
        """
        clbks = [None]
        state_manager_callback = None
        progress_bar_callback = None
        for callback in self.callbacks:
            if callback.__class__.__name__ == 'StateManagerCallback':
                state_manager_callback = callback
            elif callback.__class__.__name__ == 'ProgressBarCallback':
                progress_bar_callback = callback
            else:
                clbks.append(callback)
        clbks[0] = state_manager_callback
        clbks.append(progress_bar_callback)
        self.callbacks = clbks

    def _build_callbacks(self):
        self.callbacks = [callback for callback in self.callbacks]
        self.rearrange_callback()
        self.reconnect_callback()

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _call_sub_hook(self, mode: str, sub_mode: str, hook: str, *args: Any, **kwargs: Any):
        hook_name = f'on_{mode}_{sub_mode}_{hook}'
        for callback in self.callbacks:
            hook_fn = getattr(callback, hook_name)
            hook_fn(*args, **kwargs)  # call callback hook function, still confuse what parameter need to pass

    def _call_hook(self, mode: str, hook: str, *args: Any, **kwargs: Any):
        hook_name = f'on_{mode}_{hook}'
        for callback in self.callbacks:
            hook_fn = getattr(callback, hook_name)
            hook_fn(*args, **kwargs)  # call callback hook function, still confuse what parameter need to pass

    def on_fit_begin(self, *args: Any, **kwargs: Any):
        self._call_hook(_FIT, _BEGIN, *args, **kwargs)

    def on_fit_end(self, *args: Any, **kwargs: Any):
        self._call_hook(_FIT, _END, *args, **kwargs)

    def on_resume_begin(self, *args: Any, **kwargs: Any):
        self._call_hook(_RESUME, _BEGIN, *args, **kwargs)

    def on_resume_end(self, *args: Any, **kwargs: Any):
        self._call_hook(_RESUME, _END, *args, **kwargs)

    def on_epoch_begin(self, *args: Any, **kwargs: Any):
        self._call_hook(_EPOCH, _BEGIN, *args, **kwargs)

    def on_epoch_end(self, *args: Any, **kwargs: Any):
        self._call_hook(_EPOCH, _END, *args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        self._call_hook(_TRAIN, _BEGIN, *args, **kwargs)

    def on_train_batch_begin(self, *args, **kwargs):
        self._call_sub_hook(_TRAIN, _BATCH, _BEGIN, *args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        self._call_sub_hook(_TRAIN, _BATCH, _END, *args, **kwargs)

    def on_train_forward_begin(self, *args, **kwargs):
        self._call_sub_hook(_TRAIN, _FORWARD, _BEGIN, *args, **kwargs)

    def on_train_forward_end(self, *args, **kwargs):
        self._call_sub_hook(_TRAIN, _FORWARD, _END, *args, **kwargs)

    def on_train_backward_begin(self, *args, **kwargs):
        self._call_sub_hook(_TRAIN, _BACKWARD, _BEGIN, *args, **kwargs)

    def on_train_backward_end(self, *args, **kwargs):
        self._call_sub_hook(_TRAIN, _BACKWARD, _END, *args, **kwargs)

    def on_train_end(self, *args: Any, **kwargs: Any):
        self._call_hook(_TRAIN, _END, *args, **kwargs)

    def on_validate_begin(self, *args, **kwargs):
        self._call_hook(_VALIDATE, _BEGIN, *args, **kwargs)

    def on_validate_batch_begin(self, *args, **kwargs):
        self._call_sub_hook(_VALIDATE, _BATCH, _BEGIN, *args, **kwargs)

    def on_validate_batch_end(self, *args, **kwargs):
        self._call_sub_hook(_VALIDATE, _BATCH, _END, *args, **kwargs)

    def on_validate_forward_begin(self, *args, **kwargs):
        self._call_sub_hook(_VALIDATE, _FORWARD, _BEGIN, *args, **kwargs)

    def on_validate_forward_end(self, *args, **kwargs):
        self._call_sub_hook(_VALIDATE, _FORWARD, _END, *args, **kwargs)

    def on_validate_end(self, *args: Any, **kwargs: Any):
        self._call_hook(_VALIDATE, _END, *args, **kwargs)

    def on_predict_begin(self, *args: Any, **kwargs: Any):
        self._call_hook(_PREDICT, _BEGIN, *args, **kwargs)

    def on_predict_end(self, *args: Any, **kwargs: Any):
        self._call_hook(_PREDICT, _END, *args, **kwargs)

    def on_test_begin(self, *args: Any, **kwargs: Any):
        self._call_hook(_TEST, _BEGIN, *args, **kwargs)

    def on_test_end(self, *args: Any, **kwargs: Any):
        self._call_hook(_TEST, _END, *args, **kwargs)



if __name__ == '__main__':
    # handler = CallbackHandler()
    cb = Callback()
    method = getattr(cb, 'on_epoch_begin')
    method(10, test='10')
    # print(trainer.Trainer)
