from torchwisdom.callback import Callback
from typing import *
import torch.optim as optim
from torch.optim import lr_scheduler
from torchwisdom.statemgr.manager import StateManager
import torch


class _LRSchedulerCallback(Callback):
    def __init__(self):
        super(_LRSchedulerCallback, self).__init__()
        self.statemgr: StateManager = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: lr_scheduler._LRScheduler = None

    def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler.step()


class _SchedulerCallback(Callback):
    def __init__(self):
        super(_SchedulerCallback, self).__init__()
        self.statemgr: StateManager = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: lr_scheduler.ReduceLROnPlateau = None

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        loss: torch.Tensor = kwargs.get('loss')
        self.scheduler.step(loss)


class StepLRCallback(_LRSchedulerCallback):
    def __init__(self, step_size: int, gamma: float=0.1,
                 last_epoch: int = -1, *args: Any, **kwargs: Any):
        super(StepLRCallback, self).__init__(*args, **kwargs)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler = lr_scheduler.StepLR(self.optimizer, self.step_size, self.gamma, self.last_epoch)



class MultiStepLRCallback(_LRSchedulerCallback):
    def __init__(self,  milestones: list, gamma: float = 0.1,
                 last_epcoch: int = -1, *args: Any, **kwargs: Any)->Any:
        super(MultiStepLRCallback, self).__init__(*args, **kwargs)
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epcoch

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma, self.last_epcoch)


class LambdaLRCallback(_LRSchedulerCallback):
    def __init__(self,  lr_lambda, last_epcoch: int = -1,
                 *args: Any, **kwargs: Any) -> None:
        super(LambdaLRCallback, self).__init__(*args, **kwargs)
        self.lr_lamda = lr_lambda
        self.last_epoch = last_epcoch

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda, self.last_epcoch)


class ExponentialLRCallback(_LRSchedulerCallback):
    def __init__(self,  gamma:float, last_epoch:int = -1, *args:Any, **kwargs:Any)->None:
        super(ExponentialLRCallback, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.last_epoch = last_epoch

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, self.gamma, self.last_epoch)


class CosineAnnealingLRCallback(_LRSchedulerCallback):
    def __init__(self,  T_max:float, eta_min:float = 0,
                 last_epoch:int = -1, *args:Any, **kwargs:Any)->None:
        super(CosineAnnealingLRCallback, self).__init__(*args, **kwargs)
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, self.last_epoch)


class ReduceLROnPlateauCallback(_SchedulerCallback):
    def __init__(self, mode: str = 'min', factor: float = 0.1,
                patience: int = 10, verbose: bool = False, threshold: float = 1e-4,
                threshold_mode: str = 'rel', cooldown: int = 0, min_lr: Union[float, list] = 0,
                eps: float = 1e-8, *args, **kwargs) -> Any:
        super(ReduceLROnPlateauCallback, self).__init__(*args, **kwargs)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps

    def on_fit_begin(self, *args: Any, **kwargs: Any) -> None:
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, self.mode, self.factor,
                                                        self.patience, self.verbose, self.threshold,
                                                        self.threshold_mode, self.cooldown, self.min_lr, self.eps)




if __name__ == '__main__':
    clr = _LRSchedulerCallback()

