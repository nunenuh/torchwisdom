from torchwisdom.callback.callback import Callback
from typing import *
from torchwisdom.trainer import Trainer
import torch.optim as optim
optim.lr_scheduler.ReduceLROnPlateau

class LRSchedulerCallback(Callback):

    def step(self):
        pass

    def on_epoch_begin(self, trainer: Trainer):
        pass


class ReduceLROnPlateauCallback(Callback):

    def step(self):
        pass


if __name__ == '__main__':
    clr = LRSchedulerCallback()

