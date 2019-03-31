from torchwisdom.callback.callback import Callback
from typing import *
from torchwisdom.trainer import Trainer

class LRSchedulerCallback(Callback):

    def step(self):
        pass

    def on_epoch_begin(self, trainer: Trainer):
        pass


if __name__ == '__main__':
    clr = LRSchedulerCallback()

