from torchwisdom.callback.callback import Callback
from torchwisdom.metrics.metrics import AverageMetrics
from typing import *

class AverageMetricsCallback(Callback):
    def __init__(self):
        self.avm = AverageMetrics()

    def on_epoch_begin(self, **kwargs: Any):
        self.avm.restart()

    def on_batch_begin(self, val):
        self.avm.update(val)
