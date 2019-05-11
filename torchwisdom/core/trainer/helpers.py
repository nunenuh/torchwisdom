from collections import OrderedDict
from torchwisdom.core.callback import *


def build_default_callback_handler(trainer):
    handler: CallbackHandler = CallbackHandler(trainer=trainer)
    default_metrics: List[Callback] = [LossCallback()]
    default_callback: List[Callback] = [StateManagerCallback(), ProgressBarCallback(),
                                        ModelCallback(), OptimizerCallback()]
    clbks: List[Callback] = default_callback + default_metrics

    if trainer.metrics is not None:
        if type(trainer.metrics) is list:
            clbks = clbks + trainer.metrics
        elif isinstance(trainer.metrics, Callback):
            clbks.append(trainer.metrics)

    if trainer.callbacks is not None:
        if type(trainer.callbacks) is list:
            clbks = clbks + trainer.callbacks
        elif isinstance(trainer.callbacks, Callback):
            clbks.append(trainer.callbacks)

    handler.add(clbks)
    handler.rearrange_callback()
    return handler


def build_resume_callback_handler(trainer):
    if not trainer.handler:
        handler: CallbackHandler = CallbackHandler(trainer=trainer)
        callbacks_odict: OrderedDict = trainer.state_manager.state.get("callbacks")
        cbs = []
        for name, callback in callbacks_odict.items():
            cbs.append(callback)
        handler.add(cbs)
        return handler
    return trainer.handler

