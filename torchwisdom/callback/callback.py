from typing import *
from dataclasses import dataclass

class Hook(object):
    def on_train_begin(self, **kwargs: Any) -> None:
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        pass

    def on_valid_begin(self, **kwargs: Any) -> None:
        pass

    def on_valid_end(self, **kwargs: Any) -> None:
        pass

    def on_batch_begin(self, **kwargs: Any) -> None:
        pass

    def on_batch_end(self, **kwargs: Any) -> None:
        pass

    def on_forward_begin(self, **kwargs: Any) -> None:
        pass

    def on_forward_end(self, **kwargs: Any) -> None:
        pass

    def on_backward_begin(self, **kwargs: Any) -> None:
        pass

    def on_backward_end(self, **kwargs: Any) -> None:
        pass

    def on_epoch_begin(self, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, **kwargs: Any) -> None:
        pass

    def on_epoch_resume(self, **kwargs: Any) -> None:
        pass

class Callback(Hook):
    """ Base for all callback class """
    pass


@dataclass
class CallbackHandler(Hook):
    trainer = None
    callbacks: Collection[Callback] = None
    metrics: Collection[Callback] = None

    def __post_init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    cb_handler = CallbackHandler()

