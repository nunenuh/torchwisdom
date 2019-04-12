from torch.nn.modules.module import Module
from torchwisdom.utils.data.collector import DatasetCollector
from torchwisdom.statemgr.manager import *
from torch.optim import Optimizer
from torchwisdom.callback import *
from torchwisdom.metrics.callback import LossCallback, AccuracyCallback
from torchwisdom.statemgr.callback import StateManagerCallback
from torchwisdom.pbar import ProgressBarCallback
from torchwisdom.optim.wrapper import *


__all__ = ['Trainer']

base_history = {
    'loss': {'val': [], 'avg': [], 'epoch': []},
    'metrics': {}
}
fit_history_conf = {
    'train': base_history,
    'valid': base_history,
    'num_epoch': 0, 'epoch': 0,
}

class Trainer(object):
    def __init__(self, data: DatasetCollector, model: Module, criterion: Module, metrics: List,
                 optimizer: Optimizer = None, callbacks: List = None, device='cpu'):


        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.device = device
        self.callbacks = callbacks
        self.optwr: OptimizerWrapper = None

        # self._build_state_manager()
        # self._build_callback_handler()

    def _build_optimizer(self, lr, **kwargs):
        self.optwr = OptimizerWrapper(self.model)
        self.optimizer = self.optwr.create(lr, self.optimizer, **kwargs)

    def _build_callback_handler(self):
        self.cb_handler: CallbackHandler = CallbackHandler(trainer=self)
        default_metrics: List[Callback] = [LossCallback()]
        default_callback: List[Callback] = [StateManagerCallback(), ProgressBarCallback()]
        clbks: List[Callback] = default_callback + default_metrics

        if self.metrics is not None:
            if type(self.metrics) is list:
                clbks = clbks + self.metrics
            elif isinstance(self.metrics, Callback):
                clbks.append(self.metrics)

        if self.callbacks is not None:
            if type(self.callbacks) is list:
                clbks = clbks + self.callbacks
            elif isinstance(self.callbacks, Callback):
                clbks.append(self.callbacks)

        self.cb_handler.add(clbks)
        self.cb_handler.rearrange_callback()

    def _build_state_manager(self):
        self.state_manager: StateManager = StateManager()

        dc_state = DataCollectorState()
        model_state = ModelState()
        metric_state = MetricState()
        opt_state = OptimizerState()
        sched_state = SchedulerState()
        trainer_state = TrainerState()
        self.state_manager.add_state([dc_state, model_state, metric_state, opt_state, sched_state, trainer_state])

    def train(self, epoch, mbar):
        return NotImplementedError()

    def validate(self, epoch, mbar):
        return NotImplementedError()

    def fit(self, epoch_num):
        return NotImplementedError()

    def predict(self):
        return NotImplementedError()

    def freeze(self):
        return NotImplementedError()

    def unfreeze(self):
        return NotImplementedError()

    def save(self):
        return NotImplementedError()

    def load(self):
        return NotImplementedError()

    def export(self):
        return NotImplementedError()

